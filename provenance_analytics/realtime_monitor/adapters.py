"""Adapters between Flowcept records, the event API, and agent feedback."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any, Protocol
from typing import runtime_checkable

from .schema import EventEnvelope, EventValidationError, SCHEMA_VERSION


def _native_record_id(record: dict[str, Any]) -> str | None:
    nested = record.get("data")
    merged = dict(nested) if isinstance(nested, dict) else {}
    merged.update(record)
    for key in ("task_id", "activity_id", "record_id", "uid", "id"):
        value = merged.get(key)
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            text = str(value).strip()
            if text:
                return text
    kind = str(
        merged.get("kind")
        or merged.get("type")
        or merged.get("record_type")
        or ""
    ).casefold()
    if kind.endswith("workflow") or (
        "workflow_id" in merged
        and not any(
            merged.get(key) is not None
            for key in ("task_id", "activity_id", "record_id")
        )
    ):
        workflow = merged.get("workflow_id", merged.get("workflowId"))
        if isinstance(workflow, dict):
            workflow = next(
                (
                    workflow.get(key)
                    for key in ("id", "uid", "workflow_id")
                    if workflow.get(key) is not None
                ),
                None,
            )
        if isinstance(workflow, (str, int)) and not isinstance(workflow, bool):
            text = str(workflow).strip()
            if text:
                return text
    return None


def _identity_value(record: dict[str, Any], field: str) -> str | None:
    nested = record.get("data")
    merged = dict(nested) if isinstance(nested, dict) else {}
    merged.update(record)
    value = merged.get(field)
    if isinstance(value, dict):
        value = next(
            (value.get(key) for key in ("id", "uid", field) if value.get(key) is not None),
            None,
        )
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        text = str(value).strip()
        return text or None
    return None


class FlowceptAdapter:
    """Convert Flowcept task/workflow records to standardized event envelopes.

    The adapter has no Flowcept runtime dependency. A Flowcept stream consumer
    or local capture hook can pass its decoded records to ``to_event``.
    """

    @staticmethod
    def to_event(
        record: dict[str, Any],
        *,
        run_id: str,
        event_id: str | None = None,
        occurred_at: str | None = None,
    ) -> EventEnvelope:
        if not isinstance(record, dict):
            raise EventValidationError("Flowcept record must be an object")
        record_id = _native_record_id(record)
        if record_id is None:
            raise EventValidationError(
                "Flowcept record has no explicit task/activity/record identifier"
            )
        if event_id is None:
            canonical = json.dumps(
                [run_id, record_id, record],
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            event_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return EventEnvelope.from_dict({
            "schema_version": SCHEMA_VERSION,
            "event_id": event_id,
            "event_type": "record.upsert",
            "run_id": run_id,
            "campaign_id": _identity_value(record, "campaign_id"),
            "workflow_id": _identity_value(record, "workflow_id"),
            "agent_id": _identity_value(record, "agent_id"),
            "source": {"system": "flowcept"},
            "source_record_id": record_id,
            "occurred_at": occurred_at,
            "payload": {"record": record},
        })


class AgentFeedbackAdapter(Protocol):
    """Transport boundary for delivering corrective instructions to agents."""

    def send(self, run_id: str, intervention: dict[str, Any]) -> None: ...


class EventSink(Protocol):
    def send_event(self, event: dict[str, Any]) -> Any: ...


@runtime_checkable
class PollableFeedbackAdapter(AgentFeedbackAdapter, Protocol):
    def receive(
        self,
        run_id: str,
        limit: int = 100,
        *,
        campaign_id: str | None = None,
        workflow_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def acknowledge(
        self,
        run_id: str,
        intervention_id: str,
        *,
        campaign_id: str | None = None,
        workflow_id: str | None = None,
        agent_id: str | None = None,
    ) -> None: ...


class CallbackFeedbackAdapter:
    """Deliver feedback through an application-provided callback."""

    def __init__(self, callback: Callable[[str, dict[str, Any]], None]) -> None:
        self._callback = callback

    def send(self, run_id: str, intervention: dict[str, Any]) -> None:
        self._callback(run_id, dict(intervention))


class SQLiteFeedbackAdapter:
    """Durable feedback queue used by agents that poll the integration API."""

    def __init__(self, audit_store: Any) -> None:
        self._audit_store = audit_store

    def send(self, run_id: str, intervention: dict[str, Any]) -> None:
        from datetime import datetime, timezone

        self._audit_store.enqueue_feedback(
            run_id,
            intervention,
            datetime.now(timezone.utc).isoformat(),
        )

    def receive(
        self,
        run_id: str,
        limit: int = 100,
        *,
        campaign_id: str | None = None,
        workflow_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._audit_store.receive_feedback(
            run_id,
            limit,
            campaign_id=campaign_id,
            workflow_id=workflow_id,
            agent_id=agent_id,
        )

    def acknowledge(
        self,
        run_id: str,
        intervention_id: str,
        *,
        campaign_id: str | None = None,
        workflow_id: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        self._audit_store.acknowledge_feedback(
            run_id,
            intervention_id,
            campaign_id=campaign_id,
            workflow_id=workflow_id,
            agent_id=agent_id,
        )
