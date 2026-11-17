"""Tests for the event API, live feedback loop, and audit trail."""
from __future__ import annotations

import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realtime_monitor import (  # noqa: E402
    EventEnvelope,
    EventValidationError,
    FlowceptAdapter,
    MonitorEngine,
)
from realtime_monitor.client import MonitorClient  # noqa: E402
from realtime_monitor.policy import create_intervention  # noqa: E402
from realtime_monitor.flowcept_jsonl import FlowceptJSONLTailAdapter  # noqa: E402
from realtime_monitor.schema import SCHEMA_VERSION  # noqa: E402
from realtime_monitor.server import handler_for  # noqa: E402
from realtime_monitor.simulated_agent import run_demo  # noqa: E402


def event_for(record, run_id="run-a", event_id=None):
    return FlowceptAdapter.to_event(
        record,
        run_id=run_id,
        event_id=event_id or f"event-{record['task_id']}",
    ).to_dict()


def ordinary_record(task_id, **fields):
    return {
        "task_id": task_id,
        "workflow_id": "workflow-a",
        "campaign_id": "campaign-a",
        "activity_id": "report",
        "subtype": "academy_action",
        "status": "FINISHED",
        **fields,
    }


def test_event_contract_rejects_missing_ids_and_unknown_types():
    with pytest.raises(EventValidationError, match="source_record_id"):
        EventEnvelope.from_dict({
            "schema_version": SCHEMA_VERSION,
            "event_id": "e1",
            "event_type": "record.upsert",
            "run_id": "r1",
            "source": {"system": "test"},
            "payload": {"record": {"task_id": "t1"}},
        })
    with pytest.raises(EventValidationError, match="unsupported event_type"):
        EventEnvelope.from_dict({
            "schema_version": SCHEMA_VERSION,
            "event_id": "e2",
            "event_type": "unknown",
            "run_id": "r1",
            "source": {"system": "test"},
            "payload": {},
        })


def test_flowcept_adapter_preserves_native_id_and_rejects_missing_identity():
    event = FlowceptAdapter.to_event(
        {"data": {"task_id": "task-7"}, "workflow_id": "wf"},
        run_id="run-1",
    )
    assert event.source_record_id == "task-7"
    assert event.source_system == "flowcept"
    assert event.to_dict()["payload"]["record"]["data"]["task_id"] == "task-7"
    with pytest.raises(EventValidationError, match="no explicit"):
        FlowceptAdapter.to_event({"some_field": "no identifier"}, run_id="run-1")


def test_flowcept_jsonl_tailer_waits_for_a_complete_appended_line(tmp_path):
    class FakeClient:
        def __init__(self):
            self.events = []
            self.received = threading.Event()

        def send_event(self, event: dict[str, Any]) -> None:
            self.events.append(event)
            self.received.set()

    path = tmp_path / "flowcept.jsonl"
    line = json.dumps({
        "task_id": "task-live",
        "workflow_id": "workflow-live",
    })
    path.write_text(line[:-2], encoding="utf-8")
    client = FakeClient()
    adapter = FlowceptJSONLTailAdapter(client, "run-live")
    stop = threading.Event()
    errors = []

    def follow():
        try:
            adapter.follow(
                path,
                poll_interval=0.01,
                from_start=True,
                stop_event=stop,
            )
        except Exception as exc:
            errors.append(exc)
            stop.set()

    thread = threading.Thread(target=follow)
    thread.start()
    try:
        assert not client.received.wait(0.05)
        with path.open("ab") as stream:
            stream.write((line[-2:] + "\n").encode("utf-8"))
        assert client.received.wait(1)
        assert client.events[0]["source_record_id"] == "task-live"
        assert errors == []
    finally:
        stop.set()
        thread.join(timeout=2)
    assert not thread.is_alive()


def test_flowcept_adapter_accepts_explicit_workflow_records():
    event = FlowceptAdapter.to_event(
        {"workflow_id": "workflow-1", "type": "workflow"},
        run_id="campaign-1",
        event_id="workflow-event",
    )
    assert event.source_record_id == "workflow-1"


def test_monitor_is_idempotent_and_rejects_event_id_reuse():
    engine = MonitorEngine()
    event = event_for(ordinary_record("one"))
    first = engine.process(event)
    duplicate = engine.process(event)
    assert first["duplicate"] is False
    assert duplicate["duplicate"] is True
    conflicting = event_for(ordinary_record("different"), event_id="event-one")
    with pytest.raises(ValueError, match="reused"):
        engine.process(conflicting)
    assert len(engine.event_history("run-a")) == 1
    engine.close()


def test_record_update_requires_matching_explicit_source_id():
    engine = MonitorEngine()
    event = event_for(ordinary_record("one"))
    event["source_record_id"] = "not-one"
    with pytest.raises(ValueError, match="does not match"):
        engine.process(event)
    assert engine.event_history("run-a") == []
    engine.close()


def test_missing_output_does_not_interrupt_running_calculation():
    engine = MonitorEngine()
    running = {
        "task_id": "calc-running",
        "workflow_id": "wf",
        "campaign_id": "campaign",
        "activity_id": "compute_ionization_energy",
        "subtype": "tool_call",
        "status": "RUNNING",
        "used": {"smiles": "CCO"},
        "generated": {},
    }
    initial = engine.process(event_for(running, event_id="calc-running"))
    assert initial["interventions"] == []
    assert engine.poll_feedback("run-a") == []

    finished = {**running, "status": "FINISHED"}
    completed = engine.process(event_for(finished, event_id="calc-finished"))
    assert any(
        item["reason_code"] == "RESULT_OUTPUT_MISSING"
        for item in completed["interventions"]
    )
    engine.close()


def test_absent_result_intervention_waits_for_run_completion():
    finding = {
        "id": "finding-1",
        "check": "claim_evidence",
        "status": "INCONCLUSIVE",
        "reason_code": "NO_MATCHING_RESULT",
        "scope": "campaign-c/workflow-w",
        "source_record_ids": ["task:line-1"],
        "evidence": [],
    }
    assert create_intervention("run-a", finding, "event-1") is None
    assert create_intervention(
        "run-a", finding, "event-2", run_completed=True
    ) is not None


def test_missing_parent_gets_safe_feedback_and_verified_resolution():
    engine = MonitorEngine()
    child = ordinary_record("child", parent_task_id="external-parent")
    first = engine.process(event_for(child))
    finding = next(
        item for item in first["new_findings"]
        if item["check"] == "parent_reference"
    )
    assert finding["status"] == "INCONCLUSIVE"
    assert finding["reason_code"] == "PARENT_ID_NOT_FOUND_IN_EXPORT"

    feedback = engine.poll_feedback("run-a")
    intervention = next(item for item in feedback if item["check"] == "parent_reference")
    assert engine.poll_feedback("run-a")[0]["intervention_id"] == (
        intervention["intervention_id"]
    )
    assert "do not infer" in intervention["instruction"]
    response_without_evidence = {
        "schema_version": SCHEMA_VERSION,
        "event_id": "agent-response-text-only",
        "event_type": "agent.response",
        "run_id": "run-a",
        "source": {"system": "simulated-agent"},
        "payload": {
            "intervention_id": intervention["intervention_id"],
            "message": "I believe the parent exists.",
        },
    }
    no_evidence_result = engine.process(response_without_evidence)
    assert no_evidence_result["response_observed"] is True
    assert no_evidence_result["response_verified"] is False
    assert engine.intervention_history("run-a")[0]["status"] == (
        "response_received_unresolved"
    )
    assert engine.poll_feedback("run-a") == []

    response_with_source = {
        "schema_version": SCHEMA_VERSION,
        "event_id": "agent-response-parent-source",
        "event_type": "agent.response",
        "run_id": "run-a",
        "source": {"system": "simulated-agent"},
        "payload": {
            "intervention_id": intervention["intervention_id"],
            "message": "Added the source parent record.",
            "record_updates": [{
                "record_id": "external-parent",
                "record": ordinary_record("external-parent"),
            }],
        },
    }
    result = engine.process(response_with_source)
    assert result["response_observed"] is True
    assert result["response_verified"] is True
    replay = engine.process(response_with_source)
    assert replay["duplicate"] is True
    assert engine.poll_feedback("run-a") == []
    audit = engine.intervention_history("run-a")
    assert audit[0]["status"] == "verified_resolved"
    assert any(
        row["transition"] == "RESOLVED_BY_SUPPORTED_EVIDENCE"
        for row in engine.finding_history("run-a")
    )
    engine.close()


def test_real_parse_failure_creates_grounded_feedback_and_requires_recovery_evidence():
    engine = MonitorEngine()
    failed = ordinary_record(
        "bad-molecule",
        activity_id="compute_ionization_energy",
        subtype="agent_tool",
        status="ERROR",
        used={"smiles": "CNC(N)=O+"},
        stderr="Parse failure for CNC(N)=O+",
    )
    first = engine.process(event_for(failed, run_id="gaied-run"))
    issue = next(f for f in first["new_findings"] if f["check"] == "candidate_input_validity")
    assert issue["reason_code"] == "INPUT_PARSE_ERROR"
    assert "CNC(N)=O+" in issue["message"]
    intervention = next(i for i in first["interventions"] if i["reason_code"] == "INPUT_PARSE_ERROR")
    assert "neutral, RDKit-valid SMILES" in intervention["instruction"]
    assert engine.poll_feedback("gaied-run")

    response = {
        "schema_version": SCHEMA_VERSION,
        "event_id": "gaied-real-correction-response",
        "event_type": "agent.response",
        "run_id": "gaied-run",
        "source": {"system": "gaied-agent"},
        "payload": {
            "intervention_id": intervention["intervention_id"],
            "message": "The neutral seed was included and the generated candidate passed RDKit validation.",
        },
    }
    observed = engine.process(response)
    assert observed["response_observed"] is True
    assert observed["response_verified"] is False

    success = ordinary_record(
        "valid-molecule",
        activity_id="compute_ionization_energy",
        subtype="agent_tool",
        status="FINISHED",
        used={"smiles": "CNC(N)=O"},
        generated={"ionization_energy": 0.42, "unit": "Ha"},
    )
    recovered = engine.process(event_for(success, run_id="gaied-run"))
    assert recovered["response_verified"] is True
    assert engine.intervention_history("gaied-run")[0]["status"] == "verified_resolved"
    engine.close()


def test_live_analyzer_flags_conclusion_claim_that_conflicts_with_finite_xtb_result():
    engine = MonitorEngine()
    calc = ordinary_record(
        "real-calc",
        activity_id="compute_ionization_energy",
        subtype="agent_tool",
        used={"smiles": "CNC(N)=O"},
        generated={"ionization_energy": 14.1156, "unit": "Ha"},
    )
    engine.process(event_for(calc, run_id="claim-run"))
    bad_text = "CNC(N)=O -inf The SMILES string is invalid, preventing the simulation from running."
    conclusion = ordinary_record(
        "bad-conclusion",
        activity_id="langgraph/conclude",
        generated={
            "simulated_molecules": {"CNC(N)=O": 14.1156},
            "conclusions": [bad_text],
        },
    )
    first = engine.process(event_for(conclusion, run_id="claim-run"))
    issue = next(f for f in first["new_findings"] if f["check"] == "claim_evidence")
    assert issue["status"] == "CONTRADICTED"
    intervention = next(i for i in first["interventions"] if i["reason_code"] == "CHECK_CONFLICT")

    response = {
        "schema_version": SCHEMA_VERSION,
        "event_id": "claim-run-agent-response",
        "event_type": "agent.response",
        "run_id": "claim-run",
        "source": {"system": "gaied"},
        "payload": {
            "intervention_id": intervention["intervention_id"],
            "message": "I reviewed the analyzer evidence and am correcting my conclusion against the recorded result.",
        },
    }
    assert engine.process(response)["response_verified"] is False

    corrected = ordinary_record(
        "corrected-conclusion",
        activity_id="langgraph/conclude",
        generated={
            "simulated_molecules": {"CNC(N)=O": 14.1156},
            "conclusions": [bad_text, "CNC(N)=O has a computed ionization energy of 14.1156 Ha."],
        },
    )
    result = engine.process(event_for(corrected, run_id="claim-run"))
    assert result["response_verified"] is True
    assert engine.intervention_history("claim-run")[0]["status"] == "verified_resolved"
    engine.close()


def test_live_analyzer_detects_recorded_energy_unit_mismatch_and_intervenes():
    engine = MonitorEngine()
    calc = ordinary_record(
        "electronvolt-calc",
        activity_id="compute_ionization_energy",
        subtype="agent_tool",
        used={"smiles": "CNC(N)=O"},
        generated={"ionization_energy": 14.1156, "unit": "eV"},
    )
    engine.process(event_for(calc, run_id="unit-run"))
    conclusion = ordinary_record(
        "wrong-unit-conclusion",
        activity_id="langgraph/conclude",
        generated={
            "simulated_molecules": {"CNC(N)=O": 14.1156},
            "conclusions": ["CNC(N)=O has an ionization energy of 14.1156 kJ/mol."],
        },
    )
    result = engine.process(event_for(conclusion, run_id="unit-run"))
    finding = next(f for f in result["new_findings"] if f["reason_code"] == "UNIT_MISMATCH")
    assert finding["status"] == "CONTRADICTED"
    intervention = next(i for i in result["interventions"] if i["reason_code"] == "UNIT_MISMATCH")
    assert "recorded unit exactly" in intervention["instruction"]
    assert engine.poll_feedback("unit-run")[0]["intervention_id"] == intervention["intervention_id"]
    response = {
        "schema_version": SCHEMA_VERSION,
        "event_id": "unit-agent-response",
        "event_type": "agent.response",
        "run_id": "unit-run",
        "source": {"system": "gaied"},
        "payload": {
            "intervention_id": intervention["intervention_id"],
            "message": "I will use the eV unit recorded by the xTB task.",
        },
    }
    assert engine.process(response)["response_verified"] is False
    corrected = ordinary_record(
        "correct-unit-conclusion",
        activity_id="langgraph/conclude",
        generated={
            "simulated_molecules": {"CNC(N)=O": 14.1156},
            "conclusions": ["CNC(N)=O has an ionization energy of 14.1156 eV."],
        },
    )
    assert engine.process(event_for(corrected, run_id="unit-run"))["response_verified"] is True
    engine.close()


def test_runs_are_isolated_and_never_share_calculation_results():
    engine = MonitorEngine()
    calc = {
        "task_id": "calc-1",
        "workflow_id": "shared-wf",
        "campaign_id": "shared-campaign",
        "activity_id": "compute_ionization_energy",
        "subtype": "tool_call",
        "status": "FINISHED",
        "custom_metadata": {"tool_name": "mock"},
        "used": {"smiles": "CCO"},
        "generated": {"ionization_energy": 10.4, "unit": "eV"},
    }
    engine.process(event_for(calc, run_id="run-calc"))
    claim = ordinary_record(
        "claim-1",
        workflow_id="shared-wf",
        campaign_id="shared-campaign",
        generated={"conclusion": "CCO ionization energy is 10.4 eV."},
    )
    result = engine.process(event_for(claim, run_id="run-claim"))
    claim_findings = [
        item for item in result["new_findings"]
        if item["check"] == "claim_evidence"
    ]
    assert claim_findings
    assert all(item["status"] == "INCONCLUSIVE" for item in claim_findings)
    assert all(
        item["reason_code"] != "CHECK_PASSED"
        for item in claim_findings
    )
    engine.close()


def test_event_and_finding_state_survive_restart(tmp_path):
    db_path = tmp_path / "monitor.sqlite3"
    engine = MonitorEngine(db_path)
    engine.process(event_for(
        ordinary_record("child", parent_task_id="missing"),
        event_id="persist-event",
    ))
    engine.close()

    restored = MonitorEngine(db_path)
    duplicate = restored.process(event_for(
        ordinary_record("child", parent_task_id="missing"),
        event_id="persist-event",
    ))
    assert duplicate["duplicate"] is True
    assert len(restored.event_history("run-a")) == 1
    assert len(restored.finding_history("run-a")) > 0
    assert restored.poll_feedback("run-a")
    restored.close()


def test_http_api_runs_simulated_agent_feedback_and_verification(tmp_path):
    engine = MonitorEngine(tmp_path / "http-audit.sqlite3")
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = MonitorClient(f"http://127.0.0.1:{server.server_port}")
    try:
        with urlopen(
            f"http://127.0.0.1:{server.server_port}/health",
            timeout=2,
        ) as response:
            assert json.loads(response.read())["status"] == "ok"
        with urlopen(
            f"http://127.0.0.1:{server.server_port}/",
            timeout=2,
        ) as response:
            dashboard = response.read().decode("utf-8")
            assert response.headers["Content-Type"].startswith("text/html")
            assert "Live Provenance Monitor" in dashboard
            assert "Analyzer findings and evidence" in dashboard
        result = run_demo(
            f"http://127.0.0.1:{server.server_port}",
            run_id="http-simulated-run",
        )
        assert result["response_event"]["response_verified"] is True
        assert result["audit"]["interventions"][0]["status"] == "verified_resolved"
        assert len(result["audit"]["events"]) == 3
        assert client.poll_feedback("http-simulated-run") == []

        flowcept_file = tmp_path / "flowcept-replay.jsonl"
        flowcept_file.write_text(
            json.dumps(ordinary_record("flowcept-replay-record")),
            encoding="utf-8",
        )
        tailer = FlowceptJSONLTailAdapter(client, "flowcept-replay-run")
        assert tailer.replay(flowcept_file) == 1
        replay_audit = client.get_audit("flowcept-replay-run")
        assert len(replay_audit["events"]) == 1
        assert replay_audit["events"][0]["event"]["source"]["system"] == "flowcept"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        engine.close()


def test_gaied_bridge_forwards_an_explicit_flowcept_record(tmp_path):
    """Bridge contract test only; this fixture is not a GAIED experiment."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from realtime_bridge import LiveMonitorBridge  # noqa: E402

    engine = MonitorEngine(tmp_path / "gaied-bridge-audit.sqlite3")
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    bridge = LiveMonitorBridge("campaign-bridge-test")
    bridge.base_url = f"http://127.0.0.1:{server.server_port}"
    record = ordinary_record(
        "gaied-fixture-task",
        activity_id="compute_ionization_energy",
        used={"smiles": "CNC(N)=O"},
        generated={"ionization_energy": -0.1},
    )
    try:
        bridge.publish(record)
        audit = MonitorClient(bridge.base_url).get_audit("campaign-bridge-test")
        assert len(audit["events"]) == 1
        observed = audit["events"][0]["event"]
        assert observed["source"]["system"] == "flowcept"
        assert observed["payload"]["record"]["activity_id"] == "compute_ionization_energy"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        engine.close()


def test_http_api_rejects_invalid_event_with_400(tmp_path):
    engine = MonitorEngine(tmp_path / "http-invalid.sqlite3")
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    request = Request(
        f"http://127.0.0.1:{server.server_port}/v1/events",
        data=b'{"schema_version":"wrong"}',
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=2)
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        engine.close()
