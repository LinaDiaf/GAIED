"""Strict JSONL loading and normalization for provenance records."""
from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

KIND_ALIASES = {
    "task": "task",
    "tasks": "task",
    "activity": "task",
    "activities": "task",
    "workflow": "workflow",
    "workflows": "workflow",
    "campaign": "campaign",
    "campaigns": "campaign",
}
PARENT_KEYS = (
    "parents",
    "parent_ids",
    "parent",
    "parent_id",
    "parent_task_id",
    "parent_task_ids",
    "parent_activity_id",
    "dependencies",
    "depends_on",
)
USED_KEYS = ("used", "inputs", "input", "arguments")
GENERATED_KEYS = ("generated", "outputs", "output", "results")
KIND_KEYS = ("kind", "type", "record_type", "event_type", "entity_type")
SUCCESS_STATUSES = {"finished", "succeeded", "success", "completed", "complete", "ok"}


def _first_value(data: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _as_string(value: Any) -> Optional[str]:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    text = str(value).strip()
    return text or None


def _decode_serialized(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError):
        return value


def parse_literal(value: Any) -> Any:
    """Decode JSON/Python literals without evaluating executable code."""
    return _decode_serialized(value)


def parse_call_repr(value: Any) -> Any:
    """Decode literal arguments in reprs such as ``State(seed='x', data={})``."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    opening, closing = text.find("("), text.rfind(")")
    if opening <= 0 or closing != len(text) - 1:
        return value
    try:
        expression = ast.parse(f"_placeholder({text[opening + 1:closing]})", mode="eval")
    except (SyntaxError, ValueError):
        return value
    call = expression.body
    if not isinstance(call, ast.Call) or call.args:
        return value
    kwargs: dict[str, Any] = {}
    try:
        for item in call.keywords:
            if item.arg is None:
                return value
            kwargs[item.arg] = ast.literal_eval(item.value)
    except (ValueError, TypeError, SyntaxError):
        return value
    return kwargs


def _normalise_nested(value: Any, depth: int = 0) -> Any:
    if depth > 30:
        return value
    decoded = _decode_serialized(value)
    if decoded is value and isinstance(value, str) and re.match(r"^[A-Za-z_]\w*\s*\(", value.strip()):
        decoded = parse_call_repr(value)
    if isinstance(decoded, dict):
        return {str(key): _normalise_nested(item, depth + 1) for key, item in decoded.items()}
    if isinstance(decoded, list):
        return [_normalise_nested(item, depth + 1) for item in decoded]
    if isinstance(decoded, tuple):
        return tuple(_normalise_nested(item, depth + 1) for item in decoded)
    return decoded


def flatten(value: Any, prefix: str = ""):
    """Yield paths and leaf values from nested JSON-compatible containers."""
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten(item, path)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            yield from flatten(item, path)
    else:
        yield prefix, value


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass
class Record:
    line_no: int
    kind: str
    uid: str
    original_id: Optional[str] = None
    workflow_id: Optional[str] = None
    campaign_id: Optional[str] = None
    parents: list[str] = field(default_factory=list)
    parent_references: list[tuple[str, str]] = field(default_factory=list)
    parent_workflow_id: Optional[str] = None
    group_id: Optional[str] = None
    subtype: Optional[str] = None
    activity_id: Optional[str] = None
    name: Optional[str] = None
    status: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    tool_name: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    used: Any = None
    generated: Any = None
    raw: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    @property
    def scope(self) -> Optional[tuple[str, str]]:
        if self.campaign_id is None or self.workflow_id is None:
            return None
        return self.campaign_id, self.workflow_id

    @property
    def scope_label(self) -> str:
        if self.scope is None:
            return f"unknown-scope/line-{self.line_no}"
        return f"campaign-{self.campaign_id}/workflow-{self.workflow_id}"


@dataclass
class Dataset:
    records: list[Record] = field(default_factory=list)
    parse_errors: list[dict[str, Any]] = field(default_factory=list)
    total_lines: int = 0
    sha256: Optional[str] = None

    @property
    def tasks(self) -> list[Record]:
        return [record for record in self.records if record.kind == "task"]

    @property
    def workflow_records(self) -> list[Record]:
        return [record for record in self.records if record.kind in ("workflow", "campaign")]

    @property
    def unknown_records(self) -> list[Record]:
        return [record for record in self.records if record.kind == "unknown"]

    def tasks_by_original_id(self) -> dict[str, list[Record]]:
        result: dict[str, list[Record]] = {}
        for record in self.tasks:
            if record.original_id:
                result.setdefault(record.original_id, []).append(record)
        return result

    def workflows_by_id(self) -> dict[str, list[Record]]:
        result: dict[str, list[Record]] = {}
        for record in self.workflow_records:
            if record.workflow_id:
                result.setdefault(record.workflow_id, []).append(record)
        return result


def _record_kind(data: dict[str, Any]) -> str:
    explicit = _first_value(data, KIND_KEYS)
    if explicit is not None:
        text = str(explicit).strip().lower()
        for alias, kind in KIND_ALIASES.items():
            if text == alias or text.endswith(f".{alias}"):
                return kind
    if "workflow_id" in data and not any(key in data for key in ("task_id", "activity_id")):
        return "workflow"
    if any(key in data for key in ("task_id", "activity_id", "activity")):
        return "task"
    return "unknown"


def _extract_payload(data: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    nested = data.get("data")
    if isinstance(nested, dict):
        payload.update(nested)
    payload.update(data)
    return payload


def _extract_parent_references(data: dict[str, Any]) -> list[tuple[str, str]]:
    nested = data.get("data")
    nested = nested if isinstance(nested, dict) else {}
    references: list[tuple[str, str]] = []
    reference_keys = ("id", "uid", "task_id", "activity_id", "record_id")
    for key in PARENT_KEYS:
        if key in data:
            prefix, source = "", data
        elif key in nested:
            prefix, source = "data.", nested
        else:
            continue
        value = source[key]
        if value is None:
            continue
        items = value if isinstance(value, (list, tuple)) else [value]
        for index, item in enumerate(items):
            field_path = f"{prefix}{key}"
            if isinstance(value, (list, tuple)):
                field_path += f"[{index}]"
            if isinstance(item, dict):
                reference_key = next(
                    (candidate for candidate in reference_keys if item.get(candidate) is not None),
                    None,
                )
                if reference_key is None:
                    continue
                field_path += f".{reference_key}"
                item = item[reference_key]
            parent_id = _as_string(item)
            reference = (field_path, parent_id) if parent_id else None
            if reference and reference not in references:
                references.append(reference)
    return references


def _normalise(data: dict[str, Any], line_no: int) -> Record:
    payload = _extract_payload(data)
    metadata = payload.get("custom_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    kind = _record_kind(payload)
    original_id = _as_string(_first_value(
        payload, ("task_id", "activity_id", "record_id", "uid", "id")
    ))
    if original_id is None and kind == "workflow":
        original_id = _as_string(payload.get("workflow_id"))

    workflow_value = _first_value(payload, ("workflow_id", "workflow", "workflowId"))
    workflow_id = _as_string(workflow_value)
    if isinstance(workflow_value, dict):
        workflow_id = _as_string(_first_value(workflow_value, ("id", "uid", "workflow_id")))

    campaign_id = _as_string(_first_value(payload, ("campaign_id", "campaign", "campaignId")))
    activity_id = _as_string(_first_value(payload, ("activity_id", "activity", "activity_name")))
    status = _as_string(_first_value(payload, ("status", "state", "result_status")))
    parent_references = _extract_parent_references(data)
    parents = list(dict.fromkeys(
        reference for _, reference in parent_references
    ))
    issues = []
    if original_id is None:
        issues.append("No original record identifier was found.")
    if campaign_id is None:
        issues.append("Campaign identifier is missing.")
    if workflow_id is None:
        issues.append("Workflow identifier is missing.")
    if kind == "unknown":
        issues.append("Record kind could not be identified from its fields.")

    return Record(
        line_no=line_no,
        kind=kind,
        uid=f"{kind}:line-{line_no}",
        original_id=original_id,
        workflow_id=workflow_id,
        campaign_id=campaign_id,
        parents=parents,
        parent_references=parent_references,
        parent_workflow_id=_as_string(_first_value(payload, ("parent_workflow_id", "parentWorkflowId"))),
        group_id=_as_string(_first_value(payload, ("group_id", "group", "session_id")))
        or _as_string(_first_value(metadata, ("group_id", "group", "session_id"))),
        subtype=_as_string(_first_value(payload, ("subtype", "task_type", "activity_type"))),
        activity_id=activity_id,
        name=_as_string(_first_value(payload, ("name", "task_name", "label", "operation"))),
        status=status,
        started_at=_as_string(_first_value(payload, ("started_at", "start_time", "start_timestamp"))),
        ended_at=_as_string(_first_value(payload, ("ended_at", "end_time", "end_timestamp"))),
        tool_name=_as_string(_first_value(metadata, ("tool_name", "tool", "operation"))),
        metadata=metadata,
        used=_normalise_nested(_first_value(payload, USED_KEYS)),
        generated=_normalise_nested(_first_value(payload, GENERATED_KEYS)),
        raw=data,
        issues=issues,
    )


def _parse_lines(lines: Iterable[str], sha256: Optional[str] = None) -> Dataset:
    dataset = Dataset(sha256=sha256)
    for line_no, line in enumerate(lines, start=1):
        dataset.total_lines += 1
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            dataset.parse_errors.append({
                "line_no": line_no,
                "error": f"Invalid JSON: {exc.msg} (column {exc.colno})",
            })
            continue
        if not isinstance(decoded, dict):
            dataset.parse_errors.append({
                "line_no": line_no,
                "error": "JSONL line must contain a JSON object.",
            })
            continue
        try:
            dataset.records.append(_normalise(decoded, line_no))
        except (TypeError, ValueError, KeyError) as exc:
            dataset.parse_errors.append({
                "line_no": line_no,
                "error": f"Normalisation failed: {type(exc).__name__}: {exc}",
            })
    return dataset


def parse_lines(lines: Iterable[str]) -> Dataset:
    """Parse JSONL lines already in memory, retaining line numbers."""
    return _parse_lines(lines)


def from_dicts(records: Iterable[dict[str, Any]]) -> Dataset:
    """Build a dataset from decoded records through the same normalizer."""
    return _parse_lines(json.dumps(item, ensure_ascii=False) for item in records)


def load_jsonl(path: str | Path) -> Dataset:
    """Read a UTF-8 JSONL file; invalid byte sequences are surfaced, not replaced."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Input JSONL file not found: {file_path}")
    raw_bytes = file_path.read_bytes()
    text = raw_bytes.decode("utf-8-sig")
    return _parse_lines(text.splitlines(), hashlib.sha256(raw_bytes).hexdigest())
