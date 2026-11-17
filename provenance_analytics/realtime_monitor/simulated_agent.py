"""Deterministic mock agent for end-to-end live monitoring tests."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .adapters import FlowceptAdapter
from .client import MonitorClient
from .schema import SCHEMA_VERSION


def _calculation_record() -> dict[str, Any]:
    return {
        "task_id": "sim-calc-1",
        "workflow_id": "sim-workflow",
        "campaign_id": "sim-campaign",
        "activity_id": "compute_ionization_energy",
        "subtype": "tool_call",
        "status": "FINISHED",
        "custom_metadata": {
            "tool_name": "mock-calculator",
            "scientific_protocol": {
                "method": "MOCK-FIXTURE",
                "software": "simulated-agent",
                "software_version": "1",
                "ionization_energy_type": "vertical",
                "geometry_protocol": "neutral_optimized_cation_single_point",
                "neutral_charge": 0,
                "cation_charge": 1,
                "neutral_multiplicity": 1,
                "cation_multiplicity": 2,
                "all_geometry_optimizations_converged": True,
                "energy_unit": "eV",
                "method_parameters": {"fixture": True},
            },
        },
        "used": {"smiles": "CCO"},
        "generated": {
            "ionization_energy": 10.4,
            "unit": "eV",
        },
    }


def _claim_record(value: str) -> dict[str, Any]:
    return {
        "task_id": "sim-claim-1",
        "workflow_id": "sim-workflow",
        "campaign_id": "sim-campaign",
        "activity_id": "report",
        "subtype": "academy_action",
        "status": "FINISHED",
        "generated": {
            "conclusion": f"CCO ionization energy is {value} eV.",
        },
    }


def run_demo(base_url: str, run_id: str) -> dict[str, Any]:
    """Send a mismatch, consume feedback, respond, and check the new result."""
    client = MonitorClient(base_url)
    calc = _calculation_record()
    initial_claim = _claim_record("9.0")
    first = client.send_event(
        FlowceptAdapter.to_event(
            calc,
            run_id=run_id,
            event_id="sim-event-calc",
        ).to_dict()
    )
    second = client.send_event(
        FlowceptAdapter.to_event(
            initial_claim,
            run_id=run_id,
            event_id="sim-event-claim",
        ).to_dict()
    )
    if not any(
        finding.get("status") == "CONTRADICTED"
        and finding.get("check") == "claim_evidence"
        for finding in second["new_findings"]
    ):
        raise RuntimeError("simulator did not trigger the expected trace mismatch")

    feedback = client.poll_feedback(run_id)
    target = next(
        (item for item in feedback if item["check"] == "claim_evidence"),
        None,
    )
    if target is None:
        raise RuntimeError("simulator did not receive corrective feedback")

    response_event = {
        "schema_version": SCHEMA_VERSION,
        "event_id": "sim-event-response",
        "event_type": "agent.response",
        "run_id": run_id,
        "source": {"system": "simulated-agent"},
        "payload": {
            "intervention_id": target["intervention_id"],
            "message": (
                "Updated the mock claim to match the explicit mock calculation "
                "record for this test."
            ),
            "record_updates": [{
                "record_id": "sim-claim-1",
                "record": _claim_record("10.4"),
            }],
        },
    }
    response = client.send_event(response_event)
    if not response["response_verified"]:
        raise RuntimeError(
            "the monitor did not verify the corrected mock claim against its trace"
        )
    audit = client.get_audit(run_id)
    return {
        "first_event": first,
        "mismatch_event": second,
        "feedback": feedback,
        "response_event": response,
        "audit": audit,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send deterministic mock workflow events to a live monitor."
    )
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--run-id", default="simulated-run")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_demo(args.url, args.run_id)
    print(json.dumps({
        "run_id": args.run_id,
        "feedback_received": len(result["feedback"]),
        "response_verified": result["response_event"]["response_verified"],
        "audit_events": len(result["audit"]["events"]),
        "finding_transitions": len(result["audit"]["findings"]),
        "interventions": [
            item["status"] for item in result["audit"]["interventions"]
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
