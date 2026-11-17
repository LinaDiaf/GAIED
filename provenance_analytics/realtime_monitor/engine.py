"""Incremental, run-isolated orchestration over the existing verifier."""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from verification import SUPPORTED

from .adapters import (
    AgentFeedbackAdapter,
    PollableFeedbackAdapter,
    SQLiteFeedbackAdapter,
    _native_record_id,
)
from .audit import AuditStore
from .core import analyze_record_snapshot
from .policy import create_intervention
from .schema import EventEnvelope


@dataclass(frozen=True)
class MonitorResult:
    event_id: str
    run_id: str
    duplicate: bool
    active_finding_count: int
    new_findings: list[dict[str, Any]]
    finding_transitions: list[dict[str, Any]]
    interventions: list[dict[str, Any]]
    response_observed: bool
    response_verified: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "run_id": self.run_id,
            "duplicate": self.duplicate,
            "active_finding_count": self.active_finding_count,
            "new_findings": self.new_findings,
            "finding_transitions": self.finding_transitions,
            "interventions": self.interventions,
            "response_observed": self.response_observed,
            "response_verified": self.response_verified,
        }


class MonitorEngine:
    """Consume one event at a time and re-evaluate only its isolated run state.

    Event ingestion is incremental and idempotent. The existing deterministic
    analyzer is reused on the current run snapshot so late parent records and
    corrections can resolve earlier INCONCLUSIVE findings without cross-run
    joins.
    """

    def __init__(
        self,
        audit_path: str | Path = ":memory:",
        feedback_adapter: AgentFeedbackAdapter | None = None,
    ) -> None:
        self.audit = AuditStore(audit_path)
        self.feedback_adapter = feedback_adapter or SQLiteFeedbackAdapter(self.audit)
        self._lock = threading.RLock()
        self._records: dict[str, OrderedDict[str, dict[str, Any]]] = {}
        self._findings: dict[str, dict[str, dict[str, Any]]] = {}
        self._interventions: dict[str, dict[str, dict[str, Any]]] = {}
        for run_id, records in self.audit.load_run_records().items():
            self._records[run_id] = OrderedDict(records)
        self._findings = self.audit.load_current_findings()
        self._interventions = self.audit.load_interventions()
        self._completed_runs = self.audit.load_completed_runs()
        self._correct_legacy_unwitnessed_resolutions()
        for run_id, intervention in self.audit.pending_intervention_feedback():
            self.feedback_adapter.send(run_id, intervention)

    def _correct_legacy_unwitnessed_resolutions(self) -> None:
        """Downgrade old "resolved" states that have no agent response evidence."""
        for run_id, interventions in self._interventions.items():
            for intervention_id, intervention in interventions.items():
                if (
                    intervention.get("status") != "verified_resolved"
                    or intervention.get("response_event_id")
                ):
                    continue
                updated = {
                    **intervention,
                    "status": "superseded_by_evidence",
                    "resolution_note": (
                        "Supporting evidence arrived before any agent response; "
                        "this is not a verified correction."
                    ),
                }
                self.audit.update_intervention(
                    run_id,
                    intervention_id,
                    "superseded_by_evidence",
                    updated,
                    resolution_finding_id=updated.get("resolution_finding_id"),
                )
                interventions[intervention_id] = updated
                if isinstance(self.feedback_adapter, PollableFeedbackAdapter):
                    self.feedback_adapter.acknowledge(run_id, intervention_id)

    def close(self) -> None:
        self.audit.close()

    def process(self, event_data: EventEnvelope | dict[str, Any]) -> dict[str, Any]:
        event = (
            event_data
            if isinstance(event_data, EventEnvelope)
            else EventEnvelope.from_dict(event_data)
        )
        serialized_event = event.to_dict()
        with self._lock:
            previous = self.audit.get_event_outcome(event.event_id)
            if previous is not None:
                if previous["event"] != serialized_event:
                    raise ValueError(
                        f"event_id {event.event_id!r} was reused with different content"
                    )
                intervention_id = event.payload.get("intervention_id")
                if (
                    event.event_type == "agent.response"
                    and isinstance(intervention_id, str)
                    and intervention_id in self._interventions.get(event.run_id, {})
                    and isinstance(self.feedback_adapter, PollableFeedbackAdapter)
                ):
                    self.feedback_adapter.acknowledge(
                        event.run_id,
                        intervention_id,
                    )
                replay = dict(previous["outcome"])
                replay["duplicate"] = True
                for intervention in replay.get("interventions", []):
                    current_intervention = self._interventions.get(
                        event.run_id, {}
                    ).get(intervention.get("intervention_id"), {})
                    if current_intervention.get("status") == "awaiting_agent_response":
                        self.feedback_adapter.send(event.run_id, intervention)
                return replay

            run_records = OrderedDict(self._records.get(event.run_id, OrderedDict()))
            record_updates = self._extract_record_updates(event)
            for record_id, record in record_updates.items():
                if _native_record_id(record) != record_id:
                    raise ValueError(
                        f"record_id {record_id!r} does not match an explicit ID in its record"
                    )
                run_records[record_id] = record

            if run_records:
                findings = analyze_record_snapshot(run_records.values())
                current = {finding["id"]: finding for finding in findings}
            else:
                current = {}

            previous_findings = self._findings.get(event.run_id, {})
            new_findings = [
                finding for finding_id, finding in current.items()
                if finding_id not in previous_findings
            ]
            removed = [
                finding for finding_id, finding in previous_findings.items()
                if finding_id not in current
            ]
            new_supported = [
                finding for finding in new_findings
                if finding.get("status") == SUPPORTED
            ]
            transitions: list[dict[str, Any]] = [
                {
                    "finding_id": finding["id"],
                    "transition": "OPENED",
                    "finding": finding,
                }
                for finding in new_findings
            ]
            for finding in removed:
                resolution = self._matching_supported(finding, new_supported)
                transitions.append({
                    "finding_id": finding["id"],
                    "transition": (
                        "RESOLVED_BY_SUPPORTED_EVIDENCE"
                        if resolution else "NO_LONGER_OBSERVED"
                    ),
                    "finding": finding,
                    "resolution_finding_id": resolution.get("id") if resolution else None,
                })

            response_observed = event.event_type == "agent.response"
            target_intervention_id = (
                event.payload.get("intervention_id")
                if response_observed else None
            )
            matching_for_intervention: dict[str, dict[str, Any]] = {}
            for intervention_id, intervention in self._interventions.get(
                event.run_id, {}
            ).items():
                if intervention.get("status") not in {
                    "awaiting_agent_response", "response_received_unresolved",
                }:
                    continue
                if response_observed and intervention_id != target_intervention_id:
                    continue
                resolution = self._matching_supported(intervention, new_supported)
                if resolution:
                    matching_for_intervention[intervention_id] = resolution

            new_interventions = []
            existing_open_finding_ids = {
                item.get("finding_id")
                for item in self._interventions.get(event.run_id, {}).values()
                if item.get("status") in {
                    "awaiting_agent_response", "response_received_unresolved",
                }
            }
            run_completed = (
                event.run_id in self._completed_runs
                or event.event_type == "run.completed"
            )
            for finding in current.values():
                intervention = create_intervention(
                    event.run_id,
                    finding,
                    event.event_id,
                    run_completed=run_completed,
                )
                if (
                    intervention
                    and finding["id"] not in existing_open_finding_ids
                ):
                    new_interventions.append(intervention)
                    existing_open_finding_ids.add(finding["id"])

            intervention_updates = []
            response_verified = False
            for intervention_id, intervention in self._interventions.get(
                event.run_id, {}
            ).items():
                if intervention.get("status") not in {
                    "awaiting_agent_response", "response_received_unresolved",
                }:
                    continue
                resolution = matching_for_intervention.get(intervention_id)
                is_target = response_observed and intervention_id == target_intervention_id
                previously_responded = (
                    intervention.get("status") == "response_received_unresolved"
                    or bool(intervention.get("response_event_id"))
                )
                if resolution:
                    verified = is_target or previously_responded
                    status = "verified_resolved" if verified else "superseded_by_evidence"
                    updated = {
                        **intervention,
                        "status": status,
                        "resolved_at": _utc_now(),
                        "resolution_finding_id": resolution["id"],
                    }
                    if not verified:
                        updated["resolution_note"] = (
                            "Supporting evidence arrived before any agent response; "
                            "this is not a verified correction."
                        )
                    intervention_updates.append({
                        "intervention_id": intervention_id,
                        "status": status,
                        "intervention": updated,
                        "response_event_id": event.event_id if is_target else None,
                        "resolution_finding_id": resolution["id"],
                    })
                    if verified:
                        response_verified = True
                    elif isinstance(self.feedback_adapter, PollableFeedbackAdapter):
                        self.feedback_adapter.acknowledge(event.run_id, intervention_id)
                elif is_target:
                    updated = {
                        **intervention,
                        "status": "response_received_unresolved",
                        "response_message": event.payload["message"],
                    }
                    intervention_updates.append({
                        "intervention_id": intervention_id,
                        "status": "response_received_unresolved",
                        "intervention": updated,
                        "response_event_id": event.event_id,
                    })
                elif previously_responded:
                    updated = dict(intervention)
                    intervention_updates.append({
                        "intervention_id": intervention_id,
                        "status": intervention["status"],
                        "intervention": updated,
                    })

            result = MonitorResult(
                event_id=event.event_id,
                run_id=event.run_id,
                duplicate=False,
                active_finding_count=len(current),
                new_findings=new_findings,
                finding_transitions=transitions,
                interventions=new_interventions,
                response_observed=response_observed,
                response_verified=response_verified,
            ).to_dict()
            received_at = _utc_now()
            self.audit.save_event(
                event_id=event.event_id,
                run_id=event.run_id,
                event_type=event.event_type,
                received_at=received_at,
                event=serialized_event,
                campaign_id=event.campaign_id,
                workflow_id=event.workflow_id,
                agent_id=event.agent_id,
                outcome=result,
                record_updates=record_updates,
                current_findings=current,
                finding_transitions=transitions,
                new_interventions=new_interventions,
                intervention_updates=intervention_updates,
            )

            self._records[event.run_id] = run_records
            self._findings[event.run_id] = current
            if event.event_type == "run.completed":
                self._completed_runs.add(event.run_id)
            intervention_state = self._interventions.setdefault(event.run_id, {})
            for intervention in new_interventions:
                intervention_state[intervention["intervention_id"]] = intervention
            for update in intervention_updates:
                intervention_state[update["intervention_id"]] = update["intervention"]

            for intervention in new_interventions:
                self.feedback_adapter.send(event.run_id, intervention)
            if (
                response_observed
                and isinstance(target_intervention_id, str)
                and target_intervention_id in intervention_state
                and isinstance(self.feedback_adapter, PollableFeedbackAdapter)
            ):
                self.feedback_adapter.acknowledge(
                    event.run_id,
                    target_intervention_id,
                )
            return result

    def poll_feedback(
        self,
        run_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        adapter = self.feedback_adapter
        if not isinstance(adapter, PollableFeedbackAdapter):
            raise RuntimeError(
                "The configured feedback adapter does not support polling"
            )
        return adapter.receive(run_id, limit)

    def event_history(
        self,
        run_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return self.audit.event_history(run_id, limit, offset)

    def finding_history(
        self,
        run_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return self.audit.finding_history(run_id, limit, offset)

    def intervention_history(
        self,
        run_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return self.audit.intervention_history(run_id, limit, offset)

    @staticmethod
    def _extract_record_updates(
        event: EventEnvelope,
    ) -> dict[str, dict[str, Any]]:
        if event.event_type == "record.upsert":
            if event.source_record_id is None:
                raise ValueError("record.upsert requires source_record_id")
            record = event.payload.get("record")
            if not isinstance(record, dict):
                raise ValueError("record.upsert requires an object in payload.record")
            updates: dict[str, dict[str, Any]] = {
                event.source_record_id: record,
            }
        elif event.event_type == "agent.response":
            raw_updates = event.payload.get("record_updates", [])
            if not isinstance(raw_updates, list):
                raise ValueError("agent.response record_updates must be a list")
            updates = {}
            for index, item in enumerate(raw_updates):
                if not isinstance(item, dict):
                    raise ValueError(
                        f"record_updates[{index}] must be an object"
                    )
                record_id = item.get("record_id")
                record = item.get("record")
                if not isinstance(record_id, str) or not record_id.strip():
                    raise ValueError(
                        f"record_updates[{index}].record_id must be non-empty"
                    )
                if not isinstance(record, dict):
                    raise ValueError(
                        f"record_updates[{index}].record must be an object"
                    )
                if record_id in updates:
                    raise ValueError("agent.response has duplicate record_ids in record_updates")
                updates[record_id] = record
        else:
            updates = {}
        for record_id, record in updates.items():
            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError("record update requires a non-empty record_id")
            if not isinstance(record, dict):
                raise ValueError(f"record update {record_id!r} must be an object")
        return updates

    @staticmethod
    def _matching_supported(
        issue: dict[str, Any],
        supported_findings: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        issue_check = issue.get("check")
        issue_scope = issue.get("scope")
        source_ids = set(issue.get("source_record_ids", []))
        if not source_ids:
            return None
        for finding in supported_findings:
            finding_sources = set(finding.get("source_record_ids", []))
            if (
                finding.get("check") == issue_check
                and finding.get("scope") == issue_scope
                and (
                    source_ids.issubset(finding_sources)
                    or issue_check == "candidate_input_validity"
                )
            ):
                return finding
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_event_id(run_id: str, event_type: str, payload: dict[str, Any]) -> str:
    """Make deterministic IDs for replayable test/CLI producers."""
    data = json.dumps(
        [run_id, event_type, payload],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
