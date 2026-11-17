"""Flowcept JSONL replay and append-only file tail adapter."""
from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

from .adapters import EventSink, FlowceptAdapter
from .client import MonitorClient


class FlowceptJSONLTailAdapter:
    """Forward decoded Flowcept JSONL records to the monitor event API."""

    def __init__(self, client: EventSink, run_id: str) -> None:
        if not run_id.strip():
            raise ValueError("run_id must be non-empty")
        self.client = client
        self.run_id = run_id

    def replay(self, path: str | Path) -> int:
        file_path = Path(path).expanduser()
        with file_path.open("rb") as stream:
            data = stream.read()
        lines = data.splitlines()
        if data and not data.endswith((b"\n", b"\r")) and not lines:
            lines = [data]
        for line_number, line in enumerate(lines, start=1):
            self._send_line(line, line_number)
        return sum(bool(line.strip()) for line in lines)

    def follow(
        self,
        path: str | Path,
        *,
        poll_interval: float = 0.25,
        from_start: bool = False,
        stop_event: threading.Event | None = None,
    ) -> int:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        file_path = Path(path).expanduser()
        initial_size = file_path.stat().st_size
        offset = 0 if from_start else initial_size
        pending = bytearray()
        line_number = 0
        ingested = 0
        stop_event = stop_event or threading.Event()
        while not stop_event.is_set():
            size = file_path.stat().st_size
            if size < offset:
                offset = 0
                pending.clear()
                line_number = 0
            with file_path.open("rb") as stream:
                stream.seek(offset)
                chunk = stream.read()
                offset += len(chunk)
            pending.extend(chunk)
            while True:
                newline = pending.find(b"\n")
                if newline < 0:
                    break
                line = bytes(pending[:newline])
                del pending[:newline + 1]
                line_number += 1
                if line.strip():
                    self._send_line(line, line_number)
                    ingested += 1
            if not chunk:
                stop_event.wait(poll_interval)
        return ingested

    def _send_line(self, line: bytes, line_number: int) -> None:
        if not line.strip():
            return
        try:
            record = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid Flowcept JSONL at input line {line_number}: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(
                f"Flowcept JSONL line {line_number} must be a JSON object"
            )
        event = FlowceptAdapter.to_event(
            record,
            run_id=self.run_id,
        )
        self.client.send_event(event.to_dict())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tail or replay an append-only Flowcept JSONL stream."
    )
    parser.add_argument("--path", required=True, help="Flowcept JSONL file path.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--from-start", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Send the whole file once (including a final line without newline).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.poll_interval <= 0:
        raise SystemExit("--poll-interval must be positive")
    adapter = FlowceptJSONLTailAdapter(
        MonitorClient(args.url),
        args.run_id,
    )
    try:
        if args.replay:
            count = adapter.replay(args.path)
            print(f"Forwarded {count} Flowcept records.")
            return 0
        print(
            f"Tailing {Path(args.path).expanduser()} for run {args.run_id!r}; "
            "press Ctrl+C to stop."
        )
        count = adapter.follow(
            args.path,
            poll_interval=args.poll_interval,
            from_start=args.from_start,
        )
    except KeyboardInterrupt:
        print("\nStopping Flowcept JSONL tail.")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Forwarded {count} Flowcept records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
