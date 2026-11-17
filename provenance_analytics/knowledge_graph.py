"""Typed fact extraction and evidence-linked graph construction."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import networkx as nx

from provenance import Dataset, Record, SUCCESS_STATUSES, flatten, is_number

MOL_KEY = re.compile(r"(smiles|molecule|mol_id|inchi|compound)", re.I)
CLAIM_KEY = re.compile(r"(claim|conclusion|summary|finding|answer|report|recommendation|response|explanation)", re.I)
OBJECTIVE_KEYS = {"objective", "goal", "optimization", "optimization_direction", "direction", "user_prompt"}
OBJECTIVE_SIGNAL = re.compile(
    r"\b(maximi[sz]e|minimi[sz]e|minimum|maximum|highest|lowest|higher|lower|high|low)\b",
    re.I,
)
QUANTITY_SIGNAL = re.compile(r"\b(?:ionization|ionisation)\s+energy\b|\bie\b", re.I)
MODAL_CLAIM = re.compile(
    r"\b(?:likely|possibly|may|might|could|should|expected|hypothetical|"
    r"whether|if|perhaps|probably)\b",
    re.I,
)
COMPARATIVE_CLAIM = re.compile(
    r"\b(?:higher|lower|greater|less)\b.{0,60}\b(?:IE|ionization energy)\b.{0,60}"
    r"\b(?:than|relative to|compared with)\b|"
    r"\b(?:IE|ionization energy)\b.{0,60}\b(?:higher|lower|greater|less|more)\b"
    r".{0,60}\b(?:than|relative to|compared with)\b|"
    r"\b(?:raises?|increases?|lowers?|decreases?)\s+(?:the\s+)?"
    r"(?:IE|ionization energy)\s+(?:(?:more|less)\s+than|"
    r"(?:relative to|compared with|than))\b",
    re.I,
)
EXTREMUM_CLAIM = re.compile(
    r"\b(?:highest|lowest)\s+(?:IE|ionization energy)\b|"
    r"\b(?:IE|ionization energy)\b.{0,40}\b(?:highest|lowest)\b",
    re.I,
)
NUMERIC_CLAIM = re.compile(
    r"\b(?:ionization\s+energy|ionisation\s+energy|IE)\s*"
    r"(?:is|of|=|:)\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
    re.I,
)
RANK_KEY = re.compile(r"(?:^|_)(?:rank|ranking|rankings|ranked|leaderboard|top\d*|ordered|sorted)(?:_|$)", re.I)
GENERIC_RESULT_KEYS = {"output", "result", "results", "value", "val"}
NOT_A_RESULT = re.compile(
    r"(?:^|_)(?:id|index|idx|count|time|seed|step|steps|duration|elapsed|rank|"
    r"iteration|iter|timestamp|num|number|size|length|epoch)(?:_|$)|^n_",
    re.I,
)
IONIZATION = re.compile(r"(ionization|ionisation)(?:[_\s-]*energy)?|\bie\b", re.I)
CALCULATION_ACTIVITY = re.compile(r"(compute|calculat|simulat|xtb|ionization|ionisation|molecular)", re.I)
PROTOCOL_KEY = "scientific_protocol"


def scope_token(record: Record) -> str:
    if record.scope is None:
        return f"unknown-scope::{record.uid}"
    return json.dumps(list(record.scope), ensure_ascii=False, separators=(",", ":"))


def norm_text(text: str) -> str:
    return re.sub(r"[\s_-]+", " ", text.lower()).strip()


def mentions(text: str, name: str) -> bool:
    candidate = name.strip()
    if not candidate:
        return False
    if re.fullmatch(r"[A-Za-z0-9@+\-\[\]()=#$\\/\.]+", candidate):
        return candidate in text
    normalized = norm_text(candidate)
    if len(normalized) < 3:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])", norm_text(text)) is not None


def _leaf_name(path: str) -> str:
    return path.rsplit(".", 1)[-1].split("[", 1)[0].lower()


def _path_is_ranking(path: str) -> bool:
    return any(RANK_KEY.search(part.split("[", 1)[0]) for part in path.split("."))


def _claim_fragments(text: str, outside_scope: Optional[list[str]] = None):
    seen: set[str] = set()
    for line in text.splitlines():
        for fragment in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9*`])", line):
            candidate = fragment.strip(" \t-*#")
            if len(candidate) < 12 or not QUANTITY_SIGNAL.search(candidate):
                continue
            is_explicit_assertion = (
                COMPARATIVE_CLAIM.search(candidate)
                or EXTREMUM_CLAIM.search(candidate)
                or NUMERIC_CLAIM.search(candidate)
            )
            if (
                is_explicit_assertion
                and not MODAL_CLAIM.search(candidate)
                and not re.match(r"^(?:goal|objective|plan)\s*:", candidate, re.I)
            ):
                key = candidate.casefold()
                if key not in seen:
                    seen.add(key)
                    yield candidate
            elif outside_scope is not None:
                outside_scope.append(candidate)


def _objective_fragments(text: str):
    labeled = re.findall(
        r"\b(?:goal|objective)\s*:\s*([^\r\n]+)",
        text,
        flags=re.I,
    )
    candidates = labeled or [re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]]
    for candidate in candidates:
        candidate = candidate.strip(" \t-*#")
        if OBJECTIVE_SIGNAL.search(candidate) and QUANTITY_SIGNAL.search(candidate):
            yield candidate


@dataclass
class ResultFact:
    record_id: str
    line_no: int
    scope: str
    campaign_id: Optional[str]
    molecule: Optional[str]
    molecule_key: Optional[str]
    molecule_field: Optional[str]
    quantity: str
    value: float
    unit: Optional[str]
    field: str
    activity: Optional[str]
    tool_name: Optional[str]
    inputs_sig: str
    scientific_protocol: Optional[dict[str, Any]] = None
    protocol_field: Optional[str] = None
    protocol_error: Optional[str] = None


@dataclass
class CalcFact:
    record: Record
    scope: str
    classification: str
    molecule: Optional[str] = None
    molecule_key: Optional[str] = None
    molecule_field: Optional[str] = None
    identity_conflict: bool = False
    scientific_protocol: Optional[dict[str, Any]] = None
    protocol_field: Optional[str] = None
    protocol_error: Optional[str] = None
    results: list[ResultFact] = field(default_factory=list)
    nonfinite: list[tuple[str, Any]] = field(default_factory=list)


@dataclass
class TextFact:
    record_id: str
    line_no: int
    scope: str
    campaign_id: Optional[str]
    field: str
    text: str
    source_refs: list[tuple[str, int, str]] = dataclasses.field(default_factory=list)


@dataclass
class RankingFact:
    record_id: str
    line_no: int
    scope: str
    campaign_id: Optional[str]
    field: str
    items: list[dict[str, Any]]
    unit: Optional[str] = None
    source_refs: list[tuple[str, int]] = dataclasses.field(default_factory=list)


@dataclass
class Model:
    dataset: Dataset
    scopes: dict[str, list[Record]] = field(default_factory=dict)
    calcs: list[CalcFact] = field(default_factory=list)
    results: list[ResultFact] = field(default_factory=list)
    claims: list[TextFact] = field(default_factory=list)
    objectives: list[TextFact] = field(default_factory=list)
    rankings: list[RankingFact] = field(default_factory=list)
    claim_source_fields: int = 0
    claim_text_outside_supported_forms: int = 0
    graph: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)

    def calc_by_id(self) -> dict[str, CalcFact]:
        return {calc.record.uid: calc for calc in self.calcs}


def _strings(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, (list, tuple)):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def _identifier_scheme(key: str) -> str:
    normalized = key.casefold()
    if "smiles" in normalized:
        return "smiles"
    if "inchi" in normalized:
        return "inchi_key" if "key" in normalized else "inchi"
    return normalized


def _molecule_ids(leaves: list[tuple[str, Any]]) -> dict[str, list[tuple[str, str]]]:
    found: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path, value in leaves:
        if _path_is_ranking(path):
            continue
        key = _leaf_name(path)
        if MOL_KEY.search(key):
            found[_identifier_scheme(key)].extend((path, item) for item in _strings(value))
    return dict(found)


def _pick_molecule(
    used_ids: dict[str, list[tuple[str, str]]],
    generated_ids: dict[str, list[tuple[str, str]]],
) -> tuple[Optional[str], Optional[str], Optional[str], bool]:
    """Choose one identifier only if matching identifier fields do not conflict."""
    identifiers: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for side, grouped in (("used", used_ids), ("generated", generated_ids)):
        for key, entries in grouped.items():
            for path, value in entries:
                identifiers[key].append((side, path, value))
    if not identifiers:
        return None, None, None, False

    conflict = (
        len(identifiers) > 1
        or any(len({value for _, _, value in entries}) > 1 for entries in identifiers.values())
    )
    priority = sorted(
        identifiers,
        key=lambda key: (0 if "smiles" in key else 1 if "inchi" in key else 2, key),
    )
    key = priority[0]
    values = {value for _, _, value in identifiers[key]}
    if conflict or len(values) != 1:
        return key, None, None, True
    preferred = next(entry for entry in identifiers[key] if entry[0] == "used") if any(
        entry[0] == "used" for entry in identifiers[key]
    ) else identifiers[key][0]
    side, path, value = preferred

    return key, value, f"{side}.{path}", False


def _find_key(
    obj: Any,
    predicate: Callable[[str], bool],
    path: str = "",
    depth: int = 0,
):
    if depth > 30:
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            current = f"{path}.{key}" if path else str(key)
            if predicate(str(key)):
                yield current, value
            else:
                yield from _find_key(value, predicate, current, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            yield from _find_key(value, predicate, f"{path}[{index}]", depth + 1)


def _ranking_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict) and value and all(is_number(item) for item in value.values()):
        return [
            {"molecule": str(molecule).strip(), "molecule_key": None, "stated_value": number}
            for molecule, number in value.items()
        ]
    if not isinstance(value, (list, tuple)):
        return []
    items = []
    for entry in value:
        if (
            isinstance(entry, (list, tuple))
            and len(entry) == 2
            and isinstance(entry[0], str)
            and is_number(entry[1])
        ):
            items.append({
                "molecule": entry[0].strip(),
                "molecule_key": None,
                "stated_value": entry[1],
            })
        elif isinstance(entry, str):
            items.append({
                "molecule": entry.strip(),
                "molecule_key": None,
                "stated_value": None,
            })
        elif isinstance(entry, dict):
            molecule_entry = next(
                (
                    (path, val.strip())
                    for path, val in flatten(entry)
                    if MOL_KEY.search(_leaf_name(path)) and isinstance(val, str) and val.strip()
                ),
                (None, None),
            )
            value_pair = next(
                (
                    (path, val)
                    for path, val in flatten(entry)
                    if is_number(val) and not NOT_A_RESULT.search(_leaf_name(path))
                ),
                (None, None),
            )
            items.append({
                "molecule": molecule_entry[1],
                "molecule_key": _leaf_name(molecule_entry[0]) if molecule_entry[0] else None,
                "stated_value": value_pair[1],
            })
    return items


def _unit(generated: Any, path: str, metadata: dict[str, Any]) -> Optional[str]:
    flattened = dict(flatten(generated)) if generated is not None else {}
    parent = path.rpartition(".")[0]
    candidates = [
        f"{parent}.{name}" if parent else name
        for name in ("unit", "units", "value_unit", "value_units")
    ]
    candidates.extend(("unit", "units", "value_unit", "value_units"))
    for key in candidates:
        value = flattened.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = metadata.get("unit") or metadata.get("units")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _calculation_protocol(
    record: Record,
) -> tuple[Optional[dict[str, Any]], Optional[str], Optional[str]]:
    candidates: list[tuple[str, Any]] = []
    for side, source in (
        ("used", record.used),
        ("generated", record.generated),
        ("custom_metadata", record.metadata),
    ):
        for path, value in _find_key(
            source,
            lambda key: key.casefold() == PROTOCOL_KEY,
        ):
            candidates.append((f"{side}.{path}", value))
    if not candidates:
        return None, None, None

    field_name = candidates[0][0]
    try:
        serialized = [
            json.dumps(
                value,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            for _, value in candidates
        ]
    except (TypeError, ValueError):
        return None, field_name, "INVALID_CALCULATION_PROTOCOL"
    if len(set(serialized)) != 1:
        return None, field_name, "CONFLICTING_CALCULATION_PROTOCOL"
    if not isinstance(candidates[0][1], dict):
        return None, field_name, "INVALID_CALCULATION_PROTOCOL"
    return candidates[0][1], field_name, None


def _quantity(record: Record, path: str) -> str:
    label = " ".join(filter(None, (record.activity_id, record.tool_name))).replace("_", " ")
    name = _leaf_name(path)
    parent = path.rpartition(".")[0]
    parent_name = _leaf_name(parent) if parent else ""
    if IONIZATION.search(name) or IONIZATION.search(parent_name):
        return "ionization_energy"
    if name in GENERIC_RESULT_KEYS:
        if IONIZATION.search(label):
            return "ionization_energy"
        return parent_name if parent_name and parent_name not in GENERIC_RESULT_KEYS else "unknown"
    if IONIZATION.search(label) and name in {"energy", "ie"}:
        return "ionization_energy"
    return name


def _is_calculation(record: Record) -> bool:
    return (
        record.kind == "task"
        and (record.subtype or "").lower() in {"tool_call", "calculation", "scientific_calculation"}
        and bool(CALCULATION_ACTIVITY.search(" ".join(filter(None, (record.activity_id, record.tool_name)))))
    )


def _stable_inputs_signature(used_leaves: list[tuple[str, Any]], molecule_paths: set[str]) -> str:
    pairs = [(path, repr(value)) for path, value in used_leaves if path not in molecule_paths]
    return json.dumps(sorted(pairs), ensure_ascii=False)


def extract(dataset: Dataset) -> Model:
    model = Model(dataset=dataset)
    ranking_index: dict[tuple[str, str, Optional[str]], RankingFact] = {}
    claim_index: dict[tuple[str, str], TextFact] = {}
    for record in dataset.records:
        scope = scope_token(record)
        model.scopes.setdefault(scope, []).append(record)
        used_leaves = list(flatten(record.used)) if record.used is not None else []
        generated_leaves = list(flatten(record.generated)) if record.generated is not None else []

        seen_objectives: set[str] = set()
        for side, leaves in (("used", used_leaves), ("generated", generated_leaves)):
            for path, value in leaves:
                key = _leaf_name(path)
                if (
                    isinstance(value, str)
                    and value.strip()
                    and (
                        key in OBJECTIVE_KEYS
                        or key in {"criteria", "criterion"}
                        or re.search(r"\b(?:goal|objective)\s*:", value, re.I)
                    )
                    and not _path_is_ranking(path)
                ):
                    for objective in _objective_fragments(value.strip()):
                        dedupe_key = objective.casefold()
                        if dedupe_key not in seen_objectives:
                            seen_objectives.add(dedupe_key)
                            model.objectives.append(TextFact(
                                record.uid, record.line_no, scope, record.campaign_id,
                                f"{side}.{path}", objective,
                            ))

        seen_claims: set[str] = set()
        for path, value in generated_leaves:
            if (
                isinstance(value, str)
                and value.strip()
                and CLAIM_KEY.search(_leaf_name(path))
                and not _path_is_ranking(path)
            ):
                model.claim_source_fields += 1
                outside_scope: list[str] = []
                for claim in _claim_fragments(value.strip(), outside_scope):
                    dedupe_key = claim.casefold()
                    if dedupe_key not in seen_claims:
                        seen_claims.add(dedupe_key)
                        key = (scope, dedupe_key)
                        source_ref = (record.uid, record.line_no, f"generated.{path}")
                        existing = claim_index.get(key)
                        if existing:
                            if source_ref not in existing.source_refs:
                                existing.source_refs.append(source_ref)
                        else:
                            fact = TextFact(
                                record.uid, record.line_no, scope, record.campaign_id,
                                f"generated.{path}", claim, [source_ref],
                            )
                            claim_index[key] = fact
                            model.claims.append(fact)
                model.claim_text_outside_supported_forms += len(outside_scope)

        ranking_candidates: list[tuple[str, Any]] = []
        for side, data in (("used", record.used), ("generated", record.generated)):
            ranking_candidates.extend(
                (f"{side}.{path}", value)
                for path, value in _find_key(data, lambda key: bool(RANK_KEY.search(key)))
            )
        if (
            (record.activity_id or "").lower() == "report"
            and (record.subtype or "").lower() == "academy_action"
            and isinstance(record.generated, (list, tuple))
        ):
            ranking_candidates.append(("generated", record.generated))
        for path, value in ranking_candidates:
            items = _ranking_items(value)
            if len(items) >= 2:
                unit = _unit(record.generated, path, record.metadata)
                key = (
                    scope,
                    json.dumps(items, sort_keys=True, ensure_ascii=False),
                    unit,
                )
                existing = ranking_index.get(key)
                if existing:
                    existing.source_refs.append((record.uid, record.line_no))
                else:
                    ranking = RankingFact(
                        record.uid, record.line_no, scope, record.campaign_id, path, items,
                        unit, [(record.uid, record.line_no)],
                    )
                    ranking_index[key] = ranking
                    model.rankings.append(ranking)

        if not _is_calculation(record):
            continue

        used_ids = _molecule_ids(used_leaves)
        generated_ids = _molecule_ids(generated_leaves)
        molecule_key, molecule, molecule_field, identity_conflict = _pick_molecule(used_ids, generated_ids)
        scientific_protocol, protocol_field, protocol_error = _calculation_protocol(record)
        calc = CalcFact(
            record=record,
            scope=scope,
            classification="scientific_calculation",
            molecule=molecule,
            molecule_key=molecule_key,
            molecule_field=molecule_field,
            identity_conflict=identity_conflict,
            scientific_protocol=scientific_protocol,
            protocol_field=protocol_field,
            protocol_error=protocol_error,
        )

        molecule_paths = {
            path for values in used_ids.values() for path, _ in values
        } | {
            path for values in generated_ids.values() for path, _ in values
        }
        inputs_sig = _stable_inputs_signature(used_leaves, molecule_paths)
        completed = bool(record.status and record.status.strip().lower() in SUCCESS_STATUSES)
        if completed:
            for path, value in generated_leaves:
                if (
                    not is_number(value)
                    or _path_is_ranking(path)
                    or any(part.lower() in {"tool_calls", "stderr", "logs"} for part in path.split("."))
                    or NOT_A_RESULT.search(_leaf_name(path))
                ):
                    continue
                if not math.isfinite(float(value)):
                    calc.nonfinite.append((f"generated.{path}", value))
                    continue
                calc.results.append(ResultFact(
                    record_id=record.uid,
                    line_no=record.line_no,
                    scope=scope,
                    campaign_id=record.campaign_id,
                    molecule=molecule,
                    molecule_key=molecule_key,
                    molecule_field=molecule_field,
                    quantity=_quantity(record, path),
                    value=float(value),
                    unit=_unit(record.generated, path, record.metadata),
                    field=f"generated.{path}",
                    activity=record.activity_id,
                    tool_name=record.tool_name,
                    inputs_sig=inputs_sig,
                    scientific_protocol=scientific_protocol,
                    protocol_field=protocol_field,
                    protocol_error=protocol_error,
                ))
        model.calcs.append(calc)
        model.results.extend(calc.results)

    model.graph = build_graph(model)
    return model


def resolve_record_reference(record: Record, reference: str, dataset: Dataset) -> tuple[Optional[Record], str]:
    uid_matches = [item for item in dataset.records if item.uid == reference]
    if len(uid_matches) == 1:
        return uid_matches[0], "internal_uid"
    id_matches = [item for item in dataset.records if item.original_id == reference]
    same_scope = [item for item in id_matches if item.scope == record.scope]
    if len(same_scope) == 1:
        return same_scope[0], "same_scope_original_id"
    if len(same_scope) > 1:
        return None, "ambiguous_original_id_in_scope"
    same_campaign = [item for item in id_matches if item.campaign_id == record.campaign_id]
    if len(same_campaign) == 1:
        return same_campaign[0], "same_campaign_original_id"
    if len(id_matches) == 1:
        return id_matches[0], "unique_original_id"
    if id_matches:
        return None, "ambiguous_original_id"
    return None, "unresolved_reference"


def build_graph(model: Model) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()

    def add_node(node_id: str, node_type: str, label: str, **attrs: Any) -> None:
        if node_id not in graph:
            graph.add_node(node_id, type=node_type, label=label, **attrs)

    def add_edge(source: str, target: str, relation: str, **attrs: Any) -> None:
        if source in graph and target in graph:
            graph.add_edge(source, target, key=relation, relation=relation, **attrs)

    workflow_nodes: dict[str, str] = {}
    for scope, records in model.scopes.items():
        first = records[0]
        workflow_node = f"workflow::{scope}"
        workflow_nodes[scope] = workflow_node
        add_node(
            workflow_node, "Workflow", first.workflow_id or "(incomplete workflow scope)",
            workflow_id=first.workflow_id, campaign_id=first.campaign_id, scope=scope,
        )
        for record in records:
            evidence_node = f"evidence::{record.uid}"
            add_node(
                evidence_node, "Evidence", record.uid, record_id=record.uid,
                original_id=record.original_id, line_no=record.line_no,
                activity_id=record.activity_id, subtype=record.subtype,
                status=record.status, scope=scope, campaign_id=record.campaign_id,
            )
            add_edge(evidence_node, workflow_node, "belongs_to")

    for record in model.dataset.records:
        child_node = f"evidence::{record.uid}"
        for reference in record.parents:
            parent, method = resolve_record_reference(record, reference, model.dataset)
            if parent:
                source_fields = [
                    field_name
                    for field_name, parent_id in record.parent_references
                    if parent_id == reference
                ]
                add_edge(
                    child_node, f"evidence::{parent.uid}", "derived_from",
                    reference=reference, resolution=method,
                    source_fields=source_fields,
                )
        if record.parent_workflow_id:
            candidates = [
                item for item in model.dataset.workflow_records
                if item.workflow_id == record.parent_workflow_id
                and item.campaign_id == record.campaign_id
            ]
            if len(candidates) == 1:
                parent_scope = scope_token(candidates[0])
                add_edge(
                    workflow_nodes.get(scope_token(record), ""),
                    workflow_nodes.get(parent_scope, ""),
                    "parent_workflow",
                    parent_workflow_id=record.parent_workflow_id,
                )

    def molecule_node(result: ResultFact) -> str:
        namespace = result.campaign_id or result.scope
        scheme = result.molecule_key or "untyped"
        node_id = f"molecule::{namespace}::{scheme}::{result.molecule}"
        add_node(
            node_id, "Molecule", result.molecule or "",
            molecule=result.molecule, identifier_scheme=result.molecule_key,
            campaign_id=result.campaign_id,
        )
        return node_id

    for calc in model.calcs:
        record = calc.record
        workflow_node = workflow_nodes.get(calc.scope, "")
        evidence_node = f"evidence::{record.uid}"
        calc_node = f"calculation::{record.uid}"
        add_node(
            calc_node, "Calculation", record.activity_id or record.uid,
            record_id=record.uid, line_no=record.line_no,
            activity_id=record.activity_id, tool_name=record.tool_name,
            status=record.status, classification=calc.classification,
            scientific_protocol=calc.scientific_protocol,
            protocol_field=calc.protocol_field,
            protocol_error=calc.protocol_error,
        )
        add_edge(workflow_node, calc_node, "contains_activity")
        add_edge(calc_node, evidence_node, "supported_by")
        if calc.molecule:
            scheme = calc.molecule_key or "untyped"
            m_node = f"molecule::{record.campaign_id or calc.scope}::{scheme}::{calc.molecule}"
            add_node(
                m_node, "Molecule", calc.molecule,
                molecule=calc.molecule, identifier_scheme=calc.molecule_key,
                campaign_id=record.campaign_id,
            )
            add_edge(calc_node, m_node, "uses_molecule")
        for result in calc.results:
            result_node = f"result::{result.record_id}::{result.field}"
            add_node(
                result_node, "ScientificResult" if result.quantity != "unknown" else "NumericalOutput",
                f"{result.quantity}={result.value}", quantity=result.quantity,
                value=result.value, unit=result.unit, field=result.field,
                record_id=result.record_id, line_no=result.line_no,
                activity_id=result.activity, tool_name=result.tool_name,
                molecule_identifier_scheme=result.molecule_key,
                molecule_identifier_field=result.molecule_field,
                scientific_protocol=result.scientific_protocol,
                protocol_field=result.protocol_field,
                protocol_error=result.protocol_error,
            )
            add_edge(calc_node, result_node, "produces")
            add_edge(result_node, evidence_node, "supported_by")
            if result.molecule:
                add_edge(result_node, molecule_node(result), "about_molecule")

    for kind, facts in (("claim", model.claims), ("objective", model.objectives)):
        for fact in facts:
            refs = fact.source_refs or [(fact.record_id, fact.line_no, fact.field)]
            if kind == "claim":
                digest = hashlib.sha256(
                    f"{fact.scope}\0{fact.text}".encode("utf-8")
                ).hexdigest()
                node_id = f"claim::{digest}"
            else:
                node_id = f"objective::{fact.record_id}::{fact.field}"
            add_node(
                node_id, "ScientificClaim" if kind == "claim" else "Objective",
                fact.text[:100], text=fact.text, field=fact.field,
                record_id=fact.record_id, line_no=fact.line_no,
                campaign_id=fact.campaign_id,
                source_record_ids=[record_id for record_id, _, _ in refs],
                source_refs=[
                    {"record_id": record_id, "line_no": line_no, "field": source_field}
                    for record_id, line_no, source_field in refs
                ],
            )
            add_edge(workflow_nodes.get(fact.scope, ""), node_id, "contains")
            for record_id, _, _ in refs:
                add_edge(node_id, f"evidence::{record_id}", "supported_by")

    for ranking in model.rankings:
        node_id = f"ranking::{ranking.record_id}::{ranking.field}"
        add_node(
            node_id, "Ranking", f"Ordered results ({len(ranking.items)} items)",
            field=ranking.field, record_id=ranking.record_id,
            line_no=ranking.line_no, order=[item.get("molecule") for item in ranking.items],
            source_record_ids=[record_id for record_id, _ in ranking.source_refs],
            campaign_id=ranking.campaign_id,
        )
        add_edge(workflow_nodes.get(ranking.scope, ""), node_id, "contains")
        for record_id, _ in ranking.source_refs:
            add_edge(node_id, f"evidence::{record_id}", "supported_by")
        for position, item in enumerate(ranking.items, start=1):
            molecule = item.get("molecule")
            if molecule:
                scheme = item.get("molecule_key") or "untyped"
                molecule_id = (
                    f"molecule::{ranking.campaign_id or ranking.scope}::{scheme}::{molecule}"
                )
                add_node(
                    molecule_id, "Molecule", molecule,
                    molecule=molecule, identifier_scheme=item.get("molecule_key"),
                    campaign_id=ranking.campaign_id,
                )
                add_edge(node_id, molecule_id, "ranks_molecule", position=position)
    return graph


def graph_to_dict(graph: nx.MultiDiGraph) -> dict[str, Any]:
    return {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "nodes": [{"id": node_id, **attrs} for node_id, attrs in graph.nodes(data=True)],
        "edges": [
            {"source": source, "target": target, **attrs}
            for source, target, attrs in graph.edges(data=True)
        ],
    }
