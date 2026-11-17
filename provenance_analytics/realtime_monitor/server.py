"""Local HTTP integration API for incremental workflow events."""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from typing import Any

from .engine import MonitorEngine
from .schema import EventValidationError

MAX_REQUEST_BYTES = 5 * 1024 * 1024


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def handler_for(engine: MonitorEngine):
    class MonitorRequestHandler(BaseHTTPRequestHandler):
        server_version = "ProvenanceMonitor/1"

        def do_GET(self) -> None:
            request = urlparse(self.path)
            query = parse_qs(request.query)
            if request.path == "/":
                dashboard = Path(__file__).with_name("dashboard.html")
                body = dashboard.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if request.path == "/health":
                self._send_json(200, {
                    "status": "ok",
                    "schema_version": "1.0",
                })
                return
            run_id = query.get("run_id", [None])[0]
            if not isinstance(run_id, str) or not run_id.strip():
                self._send_json(400, {"error": "run_id query parameter is required"})
                return
            if request.path == "/v1/feedback":
                limit_value = query.get("limit", ["100"])[0]
                try:
                    limit = int(limit_value)
                except ValueError:
                    self._send_json(400, {"error": "limit must be an integer"})
                    return
                if limit < 1 or limit > 1000:
                    self._send_json(400, {"error": "limit must be between 1 and 1000"})
                    return
                self._send_json(200, {
                    "run_id": run_id,
                    "feedback": engine.poll_feedback(run_id, limit),
                })
                return
            if request.path == "/v1/audit":
                try:
                    limit = int(query.get("limit", ["100"])[0])
                    offsets = {
                        "events": int(query.get("event_offset", ["0"])[0]),
                        "findings": int(query.get("finding_offset", ["0"])[0]),
                        "interventions": int(query.get("intervention_offset", ["0"])[0]),
                    }
                except ValueError:
                    self._send_json(400, {
                        "error": "limit and stream offsets must be integers",
                    })
                    return
                if limit < 1 or limit > 1000 or min(offsets.values()) < 0:
                    self._send_json(400, {
                        "error": "limit must be 1..1000 and offsets non-negative",
                    })
                    return
                self._send_json(200, {
                    "run_id": run_id,
                    "limit": limit,
                    "offsets": offsets,
                    "events": engine.event_history(
                        run_id, limit, offsets["events"]
                    ),
                    "findings": engine.finding_history(
                        run_id, limit, offsets["findings"]
                    ),
                    "interventions": engine.intervention_history(
                        run_id, limit, offsets["interventions"]
                    ),
                })
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/v1/events":
                self._send_json(404, {"error": "not found"})
                return
            raw_length = self.headers.get("Content-Length")
            try:
                length = int(raw_length) if raw_length is not None else -1
            except ValueError:
                self._send_json(400, {"error": "invalid Content-Length"})
                return
            if length < 0:
                self._send_json(411, {"error": "Content-Length is required"})
                return
            if length > MAX_REQUEST_BYTES:
                self._send_json(413, {"error": "event payload exceeds 5 MiB limit"})
                return
            body = self.rfile.read(length)
            try:
                event = json.loads(body.decode("utf-8"))
                outcome = engine.process(event)
            except (UnicodeDecodeError, json.JSONDecodeError, EventValidationError, ValueError) as exc:
                self._send_json(400, {
                    "error": f"{type(exc).__name__}: {exc}",
                })
                return
            self._send_json(200, outcome)

        def _send_json(self, status: int, value: Any) -> None:
            body = _json_bytes(value)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            sys.stderr.write(
                f"[monitor-api] {self.address_string()} {format % args}\n"
            )

    return MonitorRequestHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Receive workflow events and serve evidence-based feedback."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--db",
        default=str(Path(__file__).resolve().parents[1] / "output" / "realtime_audit.sqlite3"),
        help="SQLite audit database path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 < args.port < 65536:
        raise SystemExit("--port must be between 1 and 65535")
    engine = MonitorEngine(args.db)
    server = ThreadingHTTPServer(
        (args.host, args.port),
        handler_for(engine),
    )
    print(f"Monitoring API listening on http://{args.host}:{args.port}")
    print(f"Audit database: {Path(args.db).expanduser().resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping monitoring API.")
    finally:
        server.server_close()
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
