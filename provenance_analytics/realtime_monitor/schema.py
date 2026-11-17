"""Versioned, framework-neutral workflow event contract."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SCHEMA_VERSION = "1.1"
SUPPORTED_SCHEMA_VERSIONS = {"1.0", SCHEMA_VERSION}
EVENT_TYPES = {
    "record.upsert",
    "agent.acknowledgement",
    "agent.response",
    "run.completed",
}


class EventValidationError(ValueError):
    """Raised when an incoming event does not match the public event contract."""


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EventValidationError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class EventEnvelope:
    """An event from an instrumented workflow or an agent feedback adapter."""

    event_id: str
    event_type: str
    run_id: str
    source_system: str
    payload: dict[str, Any]
    source_record_id: str | None = None
    campaign_id: str | None = None
    workflow_id: str | None = None
    agent_id: str | None = None
    occurred_at: str | None = None
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: Any) -> "EventEnvelope":
        if not isinstance(data, dict):
            raise EventValidationError("event must be a JSON object")
        version = data.get("schema_version")
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise EventValidationError(
                f"schema_version must be one of {sorted(SUPPORTED_SCHEMA_VERSIONS)!r}; "
                f"got {version!r}"
            )
        event_type = _required_text(data.get("event_type"), "event_type")
        if event_type not in EVENT_TYPES:
            raise EventValidationError(f"unsupported event_type: {event_type}")
        source = data.get("source")
        if not isinstance(source, dict):
            raise EventValidationError("source must be an object")
        payload = data.get("payload")
        if not isinstance(payload, dict):
            raise EventValidationError("payload must be an object")
        source_record_id = data.get("source_record_id")
        if source_record_id is not None:
            source_record_id = _required_text(source_record_id, "source_record_id")
        identity = {}
        for field_name in ("campaign_id", "workflow_id", "agent_id"):
            value = data.get(field_name)
            if value is not None:
                identity[field_name] = _required_text(value, field_name)
        occurred_at = data.get("occurred_at")
        if occurred_at is not None:
            occurred_at = _required_text(occurred_at, "occurred_at")
        event = cls(
            event_id=_required_text(data.get("event_id"), "event_id"),
            event_type=event_type,
            run_id=_required_text(data.get("run_id"), "run_id"),
            source_system=_required_text(source.get("system"), "source.system"),
            payload=payload,
            source_record_id=source_record_id,
            campaign_id=identity.get("campaign_id"),
            workflow_id=identity.get("workflow_id"),
            agent_id=identity.get("agent_id"),
            occurred_at=occurred_at,
            schema_version=version,
        )
        event.validate_payload()
        return event

    def validate_payload(self) -> None:
        if self.event_type == "record.upsert":
            if not self.source_record_id:
                raise EventValidationError(
                    "record.upsert requires source_record_id"
                )
            if not isinstance(self.payload.get("record"), dict):
                raise EventValidationError(
                    "record.upsert payload.record must be an object"
                )
        elif self.event_type == "agent.response":
            _required_text(self.payload.get("intervention_id"), "payload.intervention_id")
            _required_text(self.payload.get("message"), "payload.message")
            finding_id = self.payload.get("finding_id")
            if finding_id is not None:
                _required_text(finding_id, "payload.finding_id")
            updates = self.payload.get("record_updates", [])
            if not isinstance(updates, list):
                raise EventValidationError("payload.record_updates must be a list")
            for index, update in enumerate(updates):
                if not isinstance(update, dict):
                    raise EventValidationError(
                        f"payload.record_updates[{index}] must be an object"
                    )
                _required_text(update.get("record_id"), f"payload.record_updates[{index}].record_id")
                if not isinstance(update.get("record"), dict):
                    raise EventValidationError(
                        f"payload.record_updates[{index}].record must be an object"
                    )
        elif self.event_type == "agent.acknowledgement":
            _required_text(self.payload.get("intervention_id"), "payload.intervention_id")
            _required_text(self.payload.get("finding_id"), "payload.finding_id")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "run_id": self.run_id,
            "source": {"system": self.source_system},
            "payload": self.payload,
        }
        if self.source_record_id is not None:
            result["source_record_id"] = self.source_record_id
        for field_name in ("campaign_id", "workflow_id", "agent_id"):
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value
        if self.occurred_at is not None:
            result["occurred_at"] = self.occurred_at
        return result
