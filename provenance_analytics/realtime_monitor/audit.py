"""SQLite event, finding, intervention, and feedback audit store."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


class AuditStore:
    """Durably records the stream and the monitor's decision history."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path,
            check_same_thread=False,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                run_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                received_at TEXT NOT NULL,
                event_json TEXT NOT NULL,
                outcome_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS run_records (
                run_id TEXT NOT NULL,
                record_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                record_json TEXT NOT NULL,
                updated_by_event_id TEXT NOT NULL,
                PRIMARY KEY (run_id, record_id)
            );
            CREATE TABLE IF NOT EXISTS run_state (
                run_id TEXT PRIMARY KEY,
                completed INTEGER NOT NULL DEFAULT 0,
                completed_by_event_id TEXT,
                status TEXT NOT NULL DEFAULT 'RUNNING',
                terminal_reason TEXT,
                last_event_at TEXT,
                last_event_sequence INTEGER,
                event_count INTEGER NOT NULL DEFAULT 0,
                event_delivery_complete INTEGER NOT NULL DEFAULT 1,
                delivery_failures INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS current_findings (
                run_id TEXT NOT NULL,
                finding_id TEXT NOT NULL,
                finding_json TEXT NOT NULL,
                PRIMARY KEY (run_id, finding_id)
            );
            CREATE TABLE IF NOT EXISTS finding_history (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                finding_id TEXT NOT NULL,
                transition TEXT NOT NULL,
                finding_json TEXT NOT NULL,
                resolution_finding_id TEXT
            );
            CREATE TABLE IF NOT EXISTS interventions (
                intervention_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                finding_id TEXT NOT NULL,
                check_name TEXT NOT NULL,
                status TEXT NOT NULL,
                intervention_json TEXT NOT NULL,
                created_by_event_id TEXT NOT NULL,
                response_event_id TEXT,
                resolution_finding_id TEXT
            );
            CREATE TABLE IF NOT EXISTS feedback_queue (
                intervention_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                campaign_id TEXT,
                workflow_id TEXT,
                agent_id TEXT,
                feedback_json TEXT NOT NULL,
                status TEXT NOT NULL,
                queued_at TEXT NOT NULL,
                delivered_at TEXT,
                delivery_attempts INTEGER NOT NULL DEFAULT 0,
                last_polled_at TEXT
            );
            CREATE INDEX IF NOT EXISTS events_by_run_sequence
                ON events (run_id, sequence);
            CREATE INDEX IF NOT EXISTS finding_history_by_run_sequence
                ON finding_history (run_id, sequence);
            """
        )
        self._ensure_column("run_state", "status", "TEXT NOT NULL DEFAULT 'RUNNING'")
        self._ensure_column("run_state", "terminal_reason", "TEXT")
        self._ensure_column("run_state", "last_event_at", "TEXT")
        self._ensure_column("run_state", "last_event_sequence", "INTEGER")
        self._ensure_column("run_state", "event_count", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column(
            "run_state", "event_delivery_complete", "INTEGER NOT NULL DEFAULT 1"
        )
        self._ensure_column(
            "run_state", "delivery_failures", "INTEGER NOT NULL DEFAULT 0"
        )
        self._ensure_column("feedback_queue", "campaign_id", "TEXT")
        self._ensure_column("feedback_queue", "workflow_id", "TEXT")
        self._ensure_column("feedback_queue", "agent_id", "TEXT")
        self._ensure_column(
            "feedback_queue", "delivery_attempts", "INTEGER NOT NULL DEFAULT 0"
        )
        self._ensure_column("feedback_queue", "last_polled_at", "TEXT")
        self._connection.executescript(
            """
            CREATE INDEX IF NOT EXISTS feedback_by_recipient_status
                ON feedback_queue (
                    run_id, campaign_id, workflow_id, agent_id, status, queued_at
                );
            CREATE INDEX IF NOT EXISTS current_findings_by_run
                ON current_findings (run_id, finding_id);
            """
        )

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        columns = {
            row["name"]
            for row in self._connection.execute(f"PRAGMA table_info({table})")
        }
        if column not in columns:
            self._connection.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def get_event_outcome(self, event_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT event_json, outcome_json FROM events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "event": json.loads(row["event_json"]),
            "outcome": json.loads(row["outcome_json"]),
        }

    def load_run_records(self) -> dict[str, list[tuple[str, dict[str, Any]]]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT run_id, record_id, record_json FROM run_records "
                "ORDER BY run_id, ordinal"
            ).fetchall()
        result: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for row in rows:
            result.setdefault(row["run_id"], []).append(
                (row["record_id"], json.loads(row["record_json"]))
            )
        return result

    def load_current_findings(self) -> dict[str, dict[str, dict[str, Any]]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT run_id, finding_id, finding_json FROM current_findings"
            ).fetchall()
        result: dict[str, dict[str, dict[str, Any]]] = {}
        for row in rows:
            result.setdefault(row["run_id"], {})[row["finding_id"]] = json.loads(
                row["finding_json"]
            )
        return result

    def load_completed_runs(self) -> set[str]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT run_id FROM run_state WHERE completed = 1"
            ).fetchall()
        return {row["run_id"] for row in rows}

    def run_summary(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            state = self._connection.execute(
                "SELECT status, terminal_reason, last_event_at, last_event_sequence, "
                "event_count, event_delivery_complete, delivery_failures "
                "FROM run_state WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            active_findings = self._connection.execute(
                "SELECT COUNT(*) AS count FROM current_findings WHERE run_id = ?",
                (run_id,),
            ).fetchone()["count"]
        if state is None:
            return {
                "run_id": run_id,
                "status": "UNKNOWN",
                "terminal_reason": None,
                "last_event_at": None,
                "last_event_sequence": None,
                "event_count": 0,
                "active_finding_count": 0,
                "event_delivery_complete": True,
                "delivery_failures": 0,
            }
        return {
            "run_id": run_id,
            "status": state["status"],
            "terminal_reason": state["terminal_reason"],
            "last_event_at": state["last_event_at"],
            "last_event_sequence": state["last_event_sequence"],
            "event_count": state["event_count"],
            "active_finding_count": active_findings,
            "event_delivery_complete": bool(state["event_delivery_complete"]),
            "delivery_failures": state["delivery_failures"],
        }

    def load_interventions(self) -> dict[str, dict[str, dict[str, Any]]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT run_id, intervention_id, intervention_json, response_event_id, "
                "resolution_finding_id FROM interventions"
            ).fetchall()
        result: dict[str, dict[str, dict[str, Any]]] = {}
        for row in rows:
            intervention = json.loads(row["intervention_json"])
            if row["response_event_id"]:
                intervention["response_event_id"] = row["response_event_id"]
            if row["resolution_finding_id"]:
                intervention["resolution_finding_id"] = row["resolution_finding_id"]
            result.setdefault(row["run_id"], {})[row["intervention_id"]] = intervention
        return result

    def save_event(
        self,
        *,
        event_id: str,
        run_id: str,
        event_type: str,
        received_at: str,
        event: dict[str, Any],
        campaign_id: str | None,
        workflow_id: str | None,
        agent_id: str | None,
        outcome: dict[str, Any],
        record_updates: dict[str, dict[str, Any]],
        current_findings: dict[str, dict[str, Any]],
        finding_transitions: list[dict[str, Any]],
        new_interventions: list[dict[str, Any]],
        intervention_updates: list[dict[str, Any]],
    ) -> None:
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "INSERT INTO events "
                    "(event_id, run_id, event_type, received_at, event_json, outcome_json) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        event_id, run_id, event_type, received_at,
                        _json(event), _json(outcome),
                    ),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO run_state (run_id, completed) VALUES (?, 0)",
                    (run_id,),
                )
                sequence = connection.execute(
                    "SELECT sequence FROM events WHERE event_id = ?",
                    (event_id,),
                ).fetchone()["sequence"]
                payload = event.get("payload", {})
                delivery_complete = payload.get("event_delivery_complete", True)
                delivery_failures = payload.get("delivery_failures", 0)
                if isinstance(delivery_failures, bool) or not isinstance(delivery_failures, int):
                    delivery_failures = 0
                connection.execute(
                    "UPDATE run_state SET last_event_at = ?, last_event_sequence = ?, "
                    "event_count = event_count + 1, "
                    "event_delivery_complete = MIN(event_delivery_complete, ?), "
                    "delivery_failures = MAX(delivery_failures, ?) WHERE run_id = ?",
                    (
                        received_at,
                        sequence,
                        1 if delivery_complete is True else 0,
                        max(0, delivery_failures),
                        run_id,
                    ),
                )
                if event_type == "run.completed":
                    terminal_status = str(payload.get("status", "COMPLETED")).upper()
                    if terminal_status not in {"COMPLETED", "FAILED", "INTERRUPTED"}:
                        raise ValueError(
                            "run.completed status must be COMPLETED, FAILED, or INTERRUPTED"
                        )
                    connection.execute(
                        "UPDATE run_state SET completed = 1, completed_by_event_id = ?, "
                        "status = ?, terminal_reason = ? "
                        "WHERE run_id = ?",
                        (
                            event_id,
                            terminal_status,
                            payload.get("reason"),
                            run_id,
                        ),
                    )
                elif event_type != "agent.acknowledgement":
                    connection.execute(
                        "UPDATE run_state SET status = 'RUNNING' "
                        "WHERE run_id = ? AND completed = 0",
                        (run_id,),
                    )
                for record_id, record in record_updates.items():
                    existing = connection.execute(
                        "SELECT ordinal FROM run_records WHERE run_id = ? AND record_id = ?",
                        (run_id, record_id),
                    ).fetchone()
                    if existing:
                        ordinal = existing["ordinal"]
                    else:
                        ordinal = connection.execute(
                            "SELECT COALESCE(MAX(ordinal), -1) + 1 AS next_ordinal "
                            "FROM run_records WHERE run_id = ?",
                            (run_id,),
                        ).fetchone()["next_ordinal"]
                    connection.execute(
                        "INSERT INTO run_records "
                        "(run_id, record_id, ordinal, record_json, updated_by_event_id) "
                        "VALUES (?, ?, ?, ?, ?) "
                        "ON CONFLICT(run_id, record_id) DO UPDATE SET "
                        "record_json=excluded.record_json, "
                        "updated_by_event_id=excluded.updated_by_event_id",
                        (run_id, record_id, ordinal, _json(record), event_id),
                    )

                connection.execute(
                    "DELETE FROM current_findings WHERE run_id = ?",
                    (run_id,),
                )
                connection.executemany(
                    "INSERT INTO current_findings (run_id, finding_id, finding_json) "
                    "VALUES (?, ?, ?)",
                    [
                        (run_id, finding_id, _json(finding))
                        for finding_id, finding in current_findings.items()
                    ],
                )
                connection.executemany(
                    "INSERT INTO finding_history "
                    "(run_id, event_id, finding_id, transition, finding_json, "
                    "resolution_finding_id) VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (
                            run_id,
                            event_id,
                            item["finding_id"],
                            item["transition"],
                            _json(item["finding"]),
                            item.get("resolution_finding_id"),
                        )
                        for item in finding_transitions
                    ],
                )
                for intervention in new_interventions:
                    connection.execute(
                        "INSERT INTO interventions "
                        "(intervention_id, run_id, finding_id, check_name, status, "
                        "intervention_json, created_by_event_id) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            intervention["intervention_id"],
                            run_id,
                            intervention["finding_id"],
                            intervention["check"],
                            intervention["status"],
                            _json(intervention),
                            event_id,
                        ),
                    )
                for update in intervention_updates:
                    connection.execute(
                        "UPDATE interventions SET status = ?, intervention_json = ?, "
                        "response_event_id = COALESCE(?, response_event_id), "
                        "resolution_finding_id = COALESCE(?, resolution_finding_id) "
                        "WHERE intervention_id = ? AND run_id = ?",
                        (
                            update["status"],
                            _json(update["intervention"]),
                            update.get("response_event_id"),
                            update.get("resolution_finding_id"),
                            update["intervention_id"],
                            run_id,
                        ),
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def update_intervention(
        self,
        run_id: str,
        intervention_id: str,
        status: str,
        intervention: dict[str, Any],
        *,
        response_event_id: str | None = None,
        resolution_finding_id: str | None = None,
    ) -> None:
        with self._lock:
            self._connection.execute(
                "UPDATE interventions SET status = ?, intervention_json = ?, "
                "response_event_id = COALESCE(?, response_event_id), "
                "resolution_finding_id = COALESCE(?, resolution_finding_id) "
                "WHERE run_id = ? AND intervention_id = ?",
                (
                    status,
                    _json(intervention),
                    response_event_id,
                    resolution_finding_id,
                    run_id,
                    intervention_id,
                ),
            )

    def enqueue_feedback(
        self,
        run_id: str,
        intervention: dict[str, Any],
        queued_at: str,
    ) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT OR IGNORE INTO feedback_queue "
                "(intervention_id, run_id, campaign_id, workflow_id, agent_id, "
                "feedback_json, status, queued_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)",
                (
                    intervention["intervention_id"],
                    run_id,
                    intervention.get("campaign_id"),
                    intervention.get("workflow_id"),
                    intervention.get("agent_id"),
                    _json(intervention),
                    queued_at,
                ),
            )

    def pending_intervention_feedback(self) -> list[tuple[str, dict[str, Any]]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT run_id, intervention_json FROM interventions "
                "WHERE status = 'awaiting_agent_response' "
                "ORDER BY rowid"
            ).fetchall()
        return [
            (row["run_id"], json.loads(row["intervention_json"]))
            for row in rows
        ]

    def receive_feedback(
        self,
        run_id: str,
        limit: int,
        *,
        campaign_id: str | None = None,
        workflow_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT intervention_id, feedback_json FROM feedback_queue "
                "WHERE run_id = ? AND campaign_id IS ? AND workflow_id IS ? "
                "AND agent_id IS ? AND status = 'pending' "
                "ORDER BY queued_at, intervention_id LIMIT ?",
                (run_id, campaign_id, workflow_id, agent_id, limit),
            ).fetchall()
            if rows:
                polled_at = _utc_now()
                self._connection.executemany(
                    "UPDATE feedback_queue SET delivery_attempts = delivery_attempts + 1, "
                    "last_polled_at = ? WHERE intervention_id = ?",
                    [(polled_at, row["intervention_id"]) for row in rows],
                )
            result = []
            for row in rows:
                feedback = json.loads(row["feedback_json"])
                feedback["delivery_attempts"] = self._connection.execute(
                    "SELECT delivery_attempts FROM feedback_queue WHERE intervention_id = ?",
                    (row["intervention_id"],),
                ).fetchone()["delivery_attempts"]
                result.append(feedback)
            return result

    def acknowledge_feedback(
        self,
        run_id: str,
        intervention_id: str,
        *,
        campaign_id: str | None = None,
        workflow_id: str | None = None,
        agent_id: str | None = None,
    ) -> bool:
        with self._lock:
            cursor = self._connection.execute(
                "UPDATE feedback_queue SET status = 'acknowledged', delivered_at = ? "
                "WHERE run_id = ? AND campaign_id IS ? AND workflow_id IS ? "
                "AND agent_id IS ? AND intervention_id = ? AND status = 'pending'",
                (
                    _utc_now(),
                    run_id,
                    campaign_id,
                    workflow_id,
                    agent_id,
                    intervention_id,
                ),
            )
            return cursor.rowcount > 0

    def event_history(
        self,
        run_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        _validate_page(limit, offset)
        with self._lock:
            rows = self._connection.execute(
                "SELECT sequence, event_id, event_type, received_at, event_json, "
                "outcome_json FROM events WHERE run_id = ? ORDER BY sequence "
                "LIMIT ? OFFSET ?",
                (run_id, limit, offset),
            ).fetchall()
        return [
            {
                "sequence": row["sequence"],
                "event_id": row["event_id"],
                "event_type": row["event_type"],
                "received_at": row["received_at"],
                "event": json.loads(row["event_json"]),
                "outcome": json.loads(row["outcome_json"]),
            }
            for row in rows
        ]

    def finding_history(
        self,
        run_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        _validate_page(limit, offset)
        with self._lock:
            rows = self._connection.execute(
                "SELECT sequence, event_id, finding_id, transition, finding_json, "
                "resolution_finding_id FROM finding_history "
                "WHERE run_id = ? ORDER BY sequence LIMIT ? OFFSET ?",
                (run_id, limit, offset),
            ).fetchall()
        return [
            {
                "sequence": row["sequence"],
                "event_id": row["event_id"],
                "finding_id": row["finding_id"],
                "transition": row["transition"],
                "finding": json.loads(row["finding_json"]),
                "resolution_finding_id": row["resolution_finding_id"],
            }
            for row in rows
        ]

    def intervention_history(
        self,
        run_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        _validate_page(limit, offset)
        with self._lock:
            rows = self._connection.execute(
                "SELECT intervention_id, status, intervention_json, created_by_event_id, "
                "response_event_id, resolution_finding_id FROM interventions "
                "WHERE run_id = ? ORDER BY rowid LIMIT ? OFFSET ?",
                (run_id, limit, offset),
            ).fetchall()
        return [
            {
                "intervention_id": row["intervention_id"],
                "status": row["status"],
                "intervention": json.loads(row["intervention_json"]),
                "created_by_event_id": row["created_by_event_id"],
                "response_event_id": row["response_event_id"],
                "resolution_finding_id": row["resolution_finding_id"],
            }
            for row in rows
        ]


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _validate_page(limit: int, offset: int) -> None:
    if limit < 1 or offset < 0:
        raise ValueError("limit must be positive and offset must be non-negative")
