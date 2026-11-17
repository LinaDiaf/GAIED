"""Small HTTP bridge from the real GAIED workflow to Provenance Analytics.

The bridge is deliberately best-effort on event publication so monitor outages
do not silently change chemistry calculations. Feedback is polled only at the
agent's LangGraph plan boundary, where it is added to the real LLM prompt.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class LiveMonitorBridge:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.base_url = os.environ.get(
            "PROVENANCE_MONITOR_URL", "http://127.0.0.1:8765"
        ).rstrip("/")
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []

    @staticmethod
    def _record_id(record: dict[str, Any]) -> str | None:
        nested = record.get("data")
        merged = dict(nested) if isinstance(nested, dict) else {}
        merged.update(record)
        for key in ("task_id", "activity_id", "record_id", "uid", "id"):
            value = merged.get(key)
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                value = str(value).strip()
                if value:
                    return value
        if "workflow" in str(merged.get("type", "")).casefold():
            value = merged.get("workflow_id")
            if isinstance(value, dict):
                value = value.get("id") or value.get("uid")
            if isinstance(value, (str, int)):
                return str(value)
        return None

    def _post(self, event: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(event, ensure_ascii=False, allow_nan=False).encode()
        request = Request(
            self.base_url + "/v1/events", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urlopen(request, timeout=4) as response:
            return json.loads(response.read().decode("utf-8"))

    def publish(self, record: dict[str, Any]) -> None:
        if not isinstance(record, dict):
            return
        record_id = self._record_id(record)
        if record_id is None:
            logger.debug("Skipping Flowcept record without explicit identity")
            return
        canonical = json.dumps(
            [self.run_id, record_id, record], sort_keys=True,
            ensure_ascii=False, separators=(",", ":"), allow_nan=False,
        )
        event = {
            "schema_version": "1.0",
            "event_id": hashlib.sha256(canonical.encode()).hexdigest(),
            "event_type": "record.upsert",
            "run_id": self.run_id,
            "source": {"system": "flowcept"},
            "source_record_id": record_id,
            "payload": {"record": record},
        }
        try:
            self._post(event)
            with self._lock:
                self._records.append(dict(record))
        except Exception as exc:
            logger.warning("Live provenance event was not delivered: %s", exc)

    def poll_feedback(self) -> list[dict[str, Any]]:
        query = urlencode({"run_id": self.run_id})
        try:
            request = Request(self.base_url + "/v1/feedback?" + query)
            with urlopen(request, timeout=4) as response:
                value = json.loads(response.read().decode("utf-8"))
            return value.get("feedback", [])
        except Exception as exc:
            logger.warning("Could not poll live analyzer feedback: %s", exc)
            return []

    def response(
        self, intervention: dict[str, Any], message: str, since: int
    ) -> None:
        with self._lock:
            updates_by_id: dict[str, dict[str, Any]] = {}
            for record in self._records[since:]:
                record_id = self._record_id(record)
                if record_id:
                    updates_by_id[record_id] = record
            all_records_count = len(self._records)
        payload = {
            "intervention_id": intervention["intervention_id"],
            "message": message,
            "record_updates": [
                {"record_id": key, "record": value}
                for key, value in updates_by_id.items()
            ],
        }
        canonical = json.dumps(
            [self.run_id, "agent.response", payload], sort_keys=True,
            ensure_ascii=False, separators=(",", ":"), allow_nan=False,
        )
        event = {
            "schema_version": "1.0",
            "event_id": hashlib.sha256(canonical.encode()).hexdigest(),
            "event_type": "agent.response",
            "run_id": self.run_id,
            "source": {"system": "gaied"},
            "payload": payload,
        }
        try:
            self._post(event)
        except Exception as exc:
            logger.warning("Agent response was not delivered: %s", exc)
        logger.info(
            "Analyzer intervention %s was passed through LangGraph; %d new record(s) available for verification",
            intervention["intervention_id"], all_records_count - since,
        )


def publish_run_completed(
    run_id: str, *, status: str = "completed", reason: str | None = None
) -> None:
    bridge = LiveMonitorBridge(run_id)
    payload: dict[str, Any] = {
        "status": status,
        "reason": reason or ("GAIED campaign completed" if status == "completed" else "GAIED campaign failed"),
    }
    canonical = json.dumps([run_id, "run.completed", payload], sort_keys=True)
    event = {
        "schema_version": "1.0",
        "event_id": hashlib.sha256(canonical.encode()).hexdigest(),
        "event_type": "run.completed",
        "run_id": run_id,
        "source": {"system": "gaied"},
        "payload": payload,
    }
    try:
        bridge._post(event)
    except Exception as exc:
        logger.warning("Could not publish campaign completion: %s", exc)
