"""Conservative checks over locally recorded provenance and scientific outputs."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from knowledge_graph import (
    Model,
    RankingFact,
    ResultFact,
    graph_to_dict,
    mentions,
    norm_text,
    resolve_record_reference,
)

SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
INCONCLUSIVE = "INCONCLUSIVE"
VALID_STATUSES = {SUPPORTED, CONTRADICTED, INCONCLUSIVE}
SCIENTIFIC_ASSERTION_CHECKS = {"claim_evidence", "ranking_consistency", "numeric_consistency"}
PROTOCOL_REQUIRED_FIELDS = {
    "method",
    "software",
    "software_version",
    "ionization_energy_type",
    "geometry_protocol",
    "neutral_charge",
    "cation_charge",
    "neutral_multiplicity",
    "cation_multiplicity",
    "all_geometry_optimizations_converged",
    "energy_unit",
    "method_parameters",
}
VERTICAL_GEOMETRY_PROTOCOL = "neutral_optimized_cation_single_point"
ADIABATIC_GEOMETRY_PROTOCOL = "separately_optimized_neutral_cation"
MAX_WORDS = re.compile(r"\b(maximi[sz]e|maximal|maximum|highest|largest|greatest|higher|high)\b", re.I)
MIN_WORDS = re.compile(r"\b(minimi[sz]e|minimal|minimum|lowest|smallest|least|lower|low)\b", re.I)
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
UNIT_RE = re.compile(r"^\s*([A-Za-zµμ]+(?:\s*/\s*[A-Za-zµμ]+)?)(?![A-Za-z])")
CLAIM_VALUE_RE = re.compile(
    rf"\b(?:ionization\s+energy|ionisation\s+energy|IE)\s*"
    rf"(?:is|of|=|:)\s*(?P<value>{NUMBER_RE.pattern})",
    re.I,
)
ENERGY_UNITS = {
    "eV", "keV", "MeV", "GeV", "J", "kJ", "MJ",
    "cal", "kcal", "J/mol", "kJ/mol", "kcal/mol", "Ha",
}
IDENTIFIER_SCHEMES = {"smiles", "inchi", "inchi_key"}
INCONCLUSIVE_REASON_CODES = {
    "record_structure": "INCOMPLETE_PROVENANCE_METADATA",
    "parent_reference": "PARENT_REFERENCE_UNRESOLVED",
    "parent_workflow_reference": "PARENT_WORKFLOW_UNRESOLVED",
    "molecule_result_association": "MOLECULE_IDENTITY_AMBIGUOUS",
    "numeric_consistency": "NUMERIC_COMPARABILITY_UNRESOLVED",
    "missing_result_evidence": "RESULT_OUTPUT_MISSING",
    "nonfinite_result": "NONFINITE_RESULT",
    "input_parse_error": "INPUT_PARSE_ERROR",
}


@dataclass
class Finding:
    check: str
    status: str
    message: str
    scope: Optional[str] = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    severity: str = "info"
    affected_entities: list[str] = field(default_factory=list)
    assessment: Optional[str] = None
    reason_code: Optional[str] = None
    evidence_scope: Optional[str] = None

    def __post_init__(self):
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Invalid finding status: {self.status}")
        if self.assessment is None:
            self.assessment = (
                "performed" if self.status != INCONCLUSIVE else "not_performed"
            )
        if self.assessment not in {"performed", "not_performed", "indeterminate"}:
            raise ValueError(f"Invalid finding assessment: {self.assessment}")
        if self.reason_code is None:
            self.reason_code = (
                INCONCLUSIVE_REASON_CODES.get(self.check, "INSUFFICIENT_EVIDENCE")
                if self.status == INCONCLUSIVE else
                "CHECK_PASSED" if self.status == SUPPORTED else
                "CHECK_CONFLICT"
            )
        if self.evidence_scope is None:
            self.evidence_scope = (
                "recorded_trace_consistency"
                if self.check in SCIENTIFIC_ASSERTION_CHECKS
                else "provenance_or_protocol_completeness"
            )

    @property
    def source_record_ids(self) -> list[str]:
        return list(dict.fromkeys(
            item["record_id"]
            for item in self.evidence
            if item.get("record_id")
        ))

    @property
    def id(self) -> str:
        identity = json.dumps(
            [self.check, self.status, self.scope, self.evidence],
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["id"] = self.id
        result["source_record_ids"] = self.source_record_ids
        return result


def _evidence(record_id: Optional[str], line_no: Optional[int], field_name: str, detail: Any = None):
    item = {
        "record_id": record_id,
        "line_no": line_no,
        "field": field_name,
        "source_ref": {
            "source_type": "local_jsonl",
            "record_id": record_id,
            "line_no": line_no,
            "field": field_name,
        },
    }
    if detail is not None:
        item["detail"] = detail
    return item


def _result_evidence(result: ResultFact):
    return _evidence(
        result.record_id,
        result.line_no,
        result.field,
        {
            "quantity": result.quantity,
            "value": result.value,
            "unit": result.unit,
            "molecule": result.molecule,
            "molecule_identifier_scheme": result.molecule_key,
            "molecule_identifier_field": result.molecule_field,
            "activity": result.activity,
            "tool_name": result.tool_name,
            "scientific_protocol": result.scientific_protocol,
            "protocol_field": result.protocol_field,
            "protocol_error": result.protocol_error,
        },
    )


def _protocol_issue(result: ResultFact) -> tuple[Optional[str], list[str]]:
    protocol = result.scientific_protocol
    if result.protocol_error:
        return result.protocol_error, ["scientific_protocol"]
    if protocol is None:
        return "MISSING_CALCULATION_PROTOCOL", sorted(PROTOCOL_REQUIRED_FIELDS)

    missing = sorted(
        key for key in PROTOCOL_REQUIRED_FIELDS
        if key not in protocol or protocol[key] is None
    )
    if missing:
        return "INCOMPLETE_CALCULATION_PROTOCOL", missing

    string_fields = (
        "method",
        "software",
        "software_version",
        "geometry_protocol",
    )
    if any(
        not isinstance(protocol[key], str) or not protocol[key].strip()
        for key in string_fields
    ):
        return "INVALID_CALCULATION_PROTOCOL", [
            key for key in string_fields
            if not isinstance(protocol[key], str) or not protocol[key].strip()
        ]
    energy_type = protocol["ionization_energy_type"]
    if not isinstance(energy_type, str) or energy_type not in {"vertical", "adiabatic"}:
        return "INVALID_IONIZATION_ENERGY_TYPE", ["ionization_energy_type"]
    expected_geometry = (
        VERTICAL_GEOMETRY_PROTOCOL
        if energy_type == "vertical"
        else ADIABATIC_GEOMETRY_PROTOCOL
    )
    if protocol["geometry_protocol"] != expected_geometry:
        return "INCONSISTENT_GEOMETRY_PROTOCOL", [
            "ionization_energy_type", "geometry_protocol",
        ]
    if not isinstance(protocol["method_parameters"], dict):
        return "INVALID_CALCULATION_PROTOCOL", ["method_parameters"]
    for key in ("neutral_charge", "cation_charge"):
        value = protocol[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not float(value).is_integer()
        ):
            return "INVALID_CALCULATION_PROTOCOL", [key]
    if protocol["cation_charge"] != protocol["neutral_charge"] + 1:
        return "IONIZATION_CHARGE_MISMATCH", ["neutral_charge", "cation_charge"]
    for key in ("neutral_multiplicity", "cation_multiplicity"):
        value = protocol[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            return "INVALID_CALCULATION_PROTOCOL", [key]
    if protocol["all_geometry_optimizations_converged"] is not True:
        return "CALCULATION_NOT_CONVERGED", ["all_geometry_optimizations_converged"]
    if not isinstance(protocol["energy_unit"], str) or not protocol["energy_unit"].strip():
        return "INVALID_CALCULATION_PROTOCOL", ["energy_unit"]
    if not result.unit:
        return "MISSING_UNIT", ["unit"]
    if result.unit and protocol["energy_unit"] != result.unit:
        return "PROTOCOL_UNIT_MISMATCH", ["energy_unit"]
    if not _is_energy_unit(protocol["energy_unit"]):
        return "UNRECOGNIZED_UNIT", ["energy_unit"]
    return None, []


def _protocol_compatibility(results: list[ResultFact]) -> tuple[Optional[str], list[str]]:
    issues = [_protocol_issue(result) for result in results]
    for issue, fields in issues:
        if issue:
            return issue, fields
    signatures = {
        json.dumps(
            result.scientific_protocol,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        for result in results
    }
    if len(signatures) != 1:
        return "INCOMPARABLE_CALCULATION_PROTOCOL", []
    return None, []


def check_calculation_protocol(model: Model) -> list[Finding]:
    groups: dict[tuple[str, tuple[str, ...]], list[ResultFact]] = defaultdict(list)
    for result in model.results:
        issue, fields = _protocol_issue(result)
        if issue:
            groups[(issue, tuple(fields))].append(result)
    if not groups:
        if not model.results:
            return []
        return [Finding(
            "calculation_protocol",
            SUPPORTED,
            "All finite outputs record a complete, internally coherent calculation protocol.",
            evidence=[_result_evidence(result) for result in model.results],
            limitations=[
                "Protocol completeness does not independently validate the computational method or physical result."
            ],
            evidence_scope="protocol_completeness_only",
        )]

    findings = []
    for (issue, fields), results in sorted(groups.items()):
        evidence = []
        for result in results:
            evidence.append(_result_evidence(result))
            if result.protocol_field:
                evidence.append(_evidence(
                    result.record_id,
                    result.line_no,
                    result.protocol_field,
                    {
                        "scientific_protocol": result.scientific_protocol,
                        "protocol_error": result.protocol_error,
                    },
                ))
        contradicted = issue in {
            "IONIZATION_CHARGE_MISMATCH",
            "PROTOCOL_UNIT_MISMATCH",
            "INCONSISTENT_GEOMETRY_PROTOCOL",
        }
        findings.append(Finding(
            "calculation_protocol",
            CONTRADICTED if contradicted else INCONCLUSIVE,
            (
                f"The declared ionization-energy protocol is internally inconsistent: {issue}; "
                f"fields involved: {', '.join(fields) or 'protocol compatibility'}."
                if contradicted else
                f"The calculation protocol is not sufficiently explicit to assess the output: {issue}; "
                f"fields missing or requiring review: {', '.join(fields) or 'protocol compatibility'}."
            ),
            results[0].scope,
            evidence,
            [
                "No default method, charge, spin, geometry protocol, convergence state, or unit is inferred.",
                "A complete protocol is necessary for scientific comparison but is not independent validation.",
            ],
            severity="warning",
            affected_entities=list(dict.fromkeys(
                result.molecule for result in results if result.molecule
            )),
            reason_code=issue,
            evidence_scope="protocol_completeness_only",
        ))
    return findings


def _scope_results(model: Model) -> dict[str, list[ResultFact]]:
    grouped: dict[str, list[ResultFact]] = defaultdict(list)
    for result in model.results:
        grouped[result.scope].append(result)
    return grouped


def _direction(text: str) -> Optional[str]:
    maximum, minimum = bool(MAX_WORDS.search(text)), bool(MIN_WORDS.search(text))
    if maximum == minimum:
        return None
    return "max" if maximum else "min"


def _same_context(results: list[ResultFact]) -> bool:
    if len(results) < 2:
        return False
    scopes = {result.scope for result in results}
    quantities = {norm_text(result.quantity) for result in results}
    units = {result.unit or "" for result in results}
    contexts = {result.inputs_sig for result in results}
    activities = {norm_text(result.activity or "") for result in results}
    tools = {norm_text(result.tool_name or "") for result in results}
    identities = {_result_identity(result) for result in results}
    protocols = {
        json.dumps(
            result.scientific_protocol,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        for result in results
    }
    return (
        len(scopes) == 1
        and len(quantities) == 1
        and len(units) == 1
        and "" not in units
        and len(contexts) == 1
        and len(activities) == 1
        and len(tools) == 1
        and len(identities) == len(results)
        and None not in identities
        and len(protocols) == 1
    )


def check_association(model: Model) -> list[Finding]:
    findings = []
    for calc in model.calcs:
        if not calc.results:
            continue
        entities = [calc.molecule] if calc.molecule else []
        if not calc.molecule or calc.identity_conflict:
            findings.append(Finding(
                "molecule_result_association",
                INCONCLUSIVE,
                "Numerical output is present, but the record does not establish one unambiguous molecule identity.",
                calc.scope,
                [
                    _evidence(calc.record.uid, calc.record.line_no, "molecule_identifier", {
                        "selected_key": calc.molecule_key,
                        "conflict": calc.identity_conflict,
                    }),
                    *[_result_evidence(result) for result in calc.results],
                ],
                ["Identifiers with different types or conflicting values are not reconciled without chemical identity tooling."],
                severity="warning",
                affected_entities=entities,
            ))
            continue
        findings.append(Finding(
            "molecule_result_association",
            SUPPORTED,
            (
                "A unique typed structure identifier and numerical output are recorded in the same calculation record."
                if calc.molecule_key in IDENTIFIER_SCHEMES else
                "A unique molecule label and numerical output are recorded in the same calculation record."
            ),
            calc.scope,
            [
                _evidence(calc.record.uid, calc.record.line_no, calc.molecule_field or "molecule"),
                *[_result_evidence(result) for result in calc.results],
            ],
            ["Same-record attribution does not independently validate the chemical structure or result."],
            affected_entities=entities,
        ))
    return findings


def check_numeric_consistency(model: Model, rel_tol: float) -> list[Finding]:
    groups: dict[tuple[Any, ...], list[ResultFact]] = defaultdict(list)
    for result in model.results:
        if norm_text(result.quantity) == "unknown":
            continue
        groups[(
            result.scope,
            result.activity or "",
            result.tool_name or "",
            result.molecule or "",
            result.molecule_key or "",
            norm_text(result.quantity),
            result.inputs_sig,
            result.field,
        )].append(result)

    findings = []
    for results in groups.values():
        if len(results) < 2:
            continue
        if _result_identity(results[0]) is None:
            findings.append(Finding(
                "numeric_consistency",
                INCONCLUSIVE,
                "Repeated outputs cannot be associated with one exact typed molecular structure identifier.",
                results[0].scope,
                [_result_evidence(result) for result in results],
                ["Unresolved or untyped molecule labels are not treated as chemical identities."],
                severity="warning",
                reason_code="IDENTITY_UNRESOLVED",
            ))
            continue
        if any(not result.unit for result in results):
            findings.append(Finding(
                "numeric_consistency",
                INCONCLUSIVE,
                "Repeated outputs have no explicit recorded unit, so their scientific comparability is unknown.",
                results[0].scope,
                [_result_evidence(result) for result in results],
                ["No unit is inferred or converted."],
                severity="warning",
                affected_entities=[results[0].molecule] if results[0].molecule else [],
                reason_code="MISSING_UNIT",
            ))
            continue
        units = {_unit_key(result.unit) for result in results}
        if len(units) != 1:
            findings.append(Finding(
                "numeric_consistency",
                INCONCLUSIVE,
                "Repeated outputs use different unit codes and no conversion is available.",
                results[0].scope,
                [_result_evidence(result) for result in results],
                ["Case-sensitive unit symbols are preserved; no unit conversion is performed."],
                severity="warning",
                affected_entities=[results[0].molecule] if results[0].molecule else [],
                reason_code="UNIT_MISMATCH",
            ))
            continue
        if not _is_energy_unit(results[0].unit):
            findings.append(Finding(
                "numeric_consistency",
                INCONCLUSIVE,
                "The recorded unit is not in the supported energy-unit set, so numeric comparability is not established.",
                results[0].scope,
                [_result_evidence(result) for result in results],
                ["Units are not inferred, converted, or accepted solely because their strings match."],
                severity="warning",
                affected_entities=[results[0].molecule] if results[0].molecule else [],
                reason_code="UNRECOGNIZED_UNIT",
            ))
            continue
        protocol_issue, missing_fields = _protocol_compatibility(results)
        if protocol_issue:
            findings.append(Finding(
                "numeric_consistency",
                INCONCLUSIVE,
                f"Repeated outputs cannot be compared because their calculation protocol is incomplete or incompatible: {protocol_issue}.",
                results[0].scope,
                [_result_evidence(result) for result in results],
                [
                    "A repeated numeric value is only scientifically comparable when the explicit protocol and units agree.",
                    f"Protocol fields requiring attention: {', '.join(missing_fields) or 'protocol compatibility'}.",
                ],
                severity="warning",
                affected_entities=[results[0].molecule] if results[0].molecule else [],
                reason_code=protocol_issue,
            ))
            continue
        reference = results[0].value
        agrees = all(
            math.isclose(result.value, reference, rel_tol=rel_tol, abs_tol=0.0)
            for result in results[1:]
        )
        findings.append(Finding(
            "numeric_consistency",
            SUPPORTED if agrees else CONTRADICTED,
            (
                "Repeated outputs agree within the configured relative tolerance."
                if agrees else
                "Repeated outputs disagree despite matching recorded scope, activity, molecule, unit, and inputs."
            ),
            results[0].scope,
            [_result_evidence(result) for result in results],
            [
                "This checks internal consistency only, not physical or experimental correctness.",
                "It assumes the recorded activity and inputs capture all scientifically relevant parameters.",
            ],
            severity="warning" if not agrees else "info",
            affected_entities=[results[0].molecule] if results[0].molecule else [],
        ))
    return findings


def _claim_unit(text_after_number: str) -> Optional[str]:
    match = UNIT_RE.match(text_after_number)
    return match.group(1).strip() if match else None


def _unit_key(value: Optional[str]) -> Optional[str]:
    return value.strip() if value else None


def _is_energy_unit(value: Optional[str]) -> bool:
    return value in ENERGY_UNITS


def _mentions_quantity(text: str, quantity: str) -> bool:
    if norm_text(quantity) == "ionization energy":
        return re.search(r"\b(?:ionization|ionisation)\s+energy\b|\bie\b", text, re.I) is not None
    return mentions(text, quantity)


def _claim_quantity(text: str, results: list[ResultFact]) -> Optional[str]:
    matched = {result.quantity for result in results if _mentions_quantity(text, result.quantity)}
    return next(iter(matched)) if len(matched) == 1 else None


def _claim_evidence(claim: Any) -> list[dict[str, Any]]:
    refs = claim.source_refs or [(claim.record_id, claim.line_no, claim.field)]
    return [
        _evidence(record_id, line_no, field_name, claim.text)
        for record_id, line_no, field_name in refs
    ]


def _exact_identifier_mentions(
    text: str,
    results: list[ResultFact],
) -> list[tuple[str, str]]:
    matches: dict[tuple[str, str], int] = {}
    for result in results:
        molecule = result.molecule
        scheme = (result.molecule_key or "").lower()
        if not molecule or scheme not in IDENTIFIER_SCHEMES:
            continue
        pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(molecule)}(?![A-Za-z0-9])"
        )
        found = pattern.search(text)
        if found:
            identity = (scheme, molecule)
            matches[identity] = min(matches.get(identity, found.start()), found.start())
    return [
        identity for identity, _ in sorted(matches.items(), key=lambda item: item[1])
    ]


def _result_identity(result: ResultFact) -> Optional[tuple[str, str]]:
    scheme = (result.molecule_key or "").lower()
    if result.molecule and scheme in IDENTIFIER_SCHEMES:
        return scheme, result.molecule
    return None


def _pairwise_direction(text: str) -> Optional[str]:
    marker = re.search(r"\b(?:than|relative to|compared with)\b", text, re.I)
    if not marker:
        return None
    trend = re.search(
        r"\b(raises?|increases?|lowers?|decreases?)\s+(?:the\s+)?"
        r"(?:IE|ionization energy)\b",
        text[:marker.start()],
        re.I,
    )
    if trend:
        verb = trend.group(1).lower()
        return "greater" if verb.startswith(("raise", "increase")) else "less"
    relation_words = re.findall(
        r"\b(higher|greater|lower|less|more)\b", text[:marker.start()], re.I
    )
    if len(relation_words) != 1:
        return None
    if re.search(
        r"\b(?:but|whereas|while)\s+(?:is\s+)?(?:higher|greater|lower|less)\b",
        text[marker.end():],
        re.I,
    ):
        return None
    relation = relation_words[0].lower()
    if relation in {"higher", "greater", "more"}:
        return "greater"
    return "less"


def _unresolved_claim(
    claim: Any,
    message: str,
    reason_code: str,
    extra_evidence: Optional[list[dict[str, Any]]] = None,
    limitations: Optional[list[str]] = None,
) -> Finding:
    return Finding(
        "claim_evidence",
        INCONCLUSIVE,
        message,
        claim.scope,
        _claim_evidence(claim) + (extra_evidence or []),
        limitations or ["No chemical name-to-structure mapping or external evidence is inferred."],
        severity="warning",
        reason_code=reason_code,
    )


def check_claims(model: Model, rel_tol: float) -> list[Finding]:
    findings = []
    results_by_scope = _scope_results(model)
    for claim in model.claims:
        text = claim.text
        scope_results = results_by_scope.get(claim.scope, [])
        quantity = _claim_quantity(text, scope_results)
        if not quantity:
            findings.append(_unresolved_claim(
                claim,
                "The assertion cannot be linked to exactly one recorded scientific quantity.",
                "QUANTITY_UNRESOLVED",
            ))
            continue
        quantity_results = [
            result for result in scope_results
            if norm_text(result.quantity) == norm_text(quantity)
        ]
        if not quantity_results:
            findings.append(_unresolved_claim(
                claim,
                "No numerical result for the stated quantity is recorded in the claim's workflow scope.",
                "NO_MATCHING_RESULT",
            ))
            continue

        number_match = CLAIM_VALUE_RE.search(text)
        if number_match:
            claimed_value = float(number_match.group("value"))
            if not math.isfinite(claimed_value):
                findings.append(_unresolved_claim(
                    claim,
                    "The numeric assertion is outside the finite numeric range supported by this verifier.",
                    "NONFINITE_CLAIM_VALUE",
                ))
                continue
            stated_unit = _claim_unit(text[number_match.end():])
            identities = _exact_identifier_mentions(text, quantity_results)
            if len(identities) != 1:
                findings.append(_unresolved_claim(
                    claim,
                    "The numerical assertion does not identify exactly one recorded molecular structure identifier.",
                    "IDENTITY_UNRESOLVED",
                    limitations=[
                        "Chemical names and labels are not mapped to SMILES or InChI without an explicit, traceable mapping."
                    ],
                ))
                continue
            scheme, molecule = identities[0]
            candidates = [
                result for result in quantity_results
                if result.molecule == molecule
                and (result.molecule_key or "").lower() == scheme
            ]
            if len(candidates) != 1:
                findings.append(Finding(
                    "claim_evidence", INCONCLUSIVE,
                    "The identifier does not map to exactly one molecule-specific result.",
                    claim.scope,
                    _claim_evidence(claim) + [_result_evidence(result) for result in candidates],
                    ["Multiple result records may differ in parameters or execution context."],
                    severity="warning",
                    affected_entities=[molecule],
                    reason_code="AMBIGUOUS_RESULT",
                ))
                continue
            result = candidates[0]
            if not stated_unit or not result.unit:
                findings.append(Finding(
                    "claim_evidence", INCONCLUSIVE,
                    "The claim or recorded result omits an explicit unit.",
                    claim.scope,
                    _claim_evidence(claim) + [
                        _evidence(claim.record_id, claim.line_no, claim.field, {
                            "claim": text, "claim_unit": stated_unit,
                        }),
                        _result_evidence(result),
                    ],
                    ["No unit is assumed, converted, or inferred."],
                    severity="warning",
                    affected_entities=[molecule],
                    reason_code="MISSING_UNIT",
                ))
                continue
            if (
                not _is_energy_unit(stated_unit)
                or not _is_energy_unit(result.unit)
            ):
                findings.append(Finding(
                    "claim_evidence", INCONCLUSIVE,
                    "The unit is not in the supported energy-unit set.",
                    claim.scope,
                    _claim_evidence(claim) + [_result_evidence(result)],
                    ["No unknown unit token is accepted or converted."],
                    severity="warning",
                    affected_entities=[molecule],
                    reason_code="UNRECOGNIZED_UNIT",
                ))
                continue
            if _unit_key(stated_unit) != _unit_key(result.unit):
                findings.append(Finding(
                    "claim_evidence", INCONCLUSIVE,
                    "The claim and recorded result use different unit codes.",
                    claim.scope,
                    _claim_evidence(claim) + [_result_evidence(result)],
                    ["No unit conversion is performed; case-sensitive unit symbols are preserved."],
                    severity="warning",
                    affected_entities=[molecule],
                    reason_code="UNIT_MISMATCH",
                ))
                continue
            protocol_issue, missing_fields = _protocol_compatibility([result])
            if protocol_issue:
                findings.append(Finding(
                    "claim_evidence", INCONCLUSIVE,
                    f"The numeric claim matches a recorded output only if its calculation protocol is established; required protocol evidence is missing or inconsistent: {protocol_issue}.",
                    claim.scope,
                    _claim_evidence(claim) + [_result_evidence(result)],
                    [
                        "Agreement with one recorded value does not independently validate the computed scientific quantity.",
                        f"Protocol fields requiring attention: {', '.join(missing_fields) or 'protocol consistency'}.",
                    ],
                    severity="warning",
                    affected_entities=[molecule],
                    reason_code=protocol_issue,
                ))
                continue
            agrees = math.isclose(claimed_value, result.value, rel_tol=rel_tol, abs_tol=0.0)
            findings.append(Finding(
                "claim_evidence",
                SUPPORTED if agrees else CONTRADICTED,
                (
                    "The stated value and explicit unit agree with the uniquely matched recorded output."
                    if agrees else
                    "The stated value conflicts with the uniquely matched recorded output."
                ),
                claim.scope,
                _claim_evidence(claim) + [_result_evidence(result)],
                [
                    "This establishes agreement with a trace, not scientific truth.",
                    "The configured tolerance is not a model of claim rounding or measurement uncertainty.",
                ],
                severity="warning" if not agrees else "info",
                affected_entities=[molecule],
            ))
            continue

        pairwise_direction = _pairwise_direction(text)
        if pairwise_direction:
            identities = _exact_identifier_mentions(text, quantity_results)
            if len(identities) < 2:
                limitations = [
                    "Chemical names and labels are not mapped to structure identifiers."
                ]
                blockers = ["IDENTITY_UNRESOLVED"]
                if any(not result.unit for result in quantity_results):
                    limitations.append(
                        "The same-scope calculation results also omit explicit units."
                    )
                    blockers.append("MISSING_UNIT")
                protocol_issue, _ = _protocol_compatibility(quantity_results)
                if quantity_results and protocol_issue:
                    limitations.append(
                        f"The same-scope outputs also lack a compatible explicit protocol ({protocol_issue})."
                    )
                    blockers.append(protocol_issue)
                blocker_text = (
                    f" Other blockers: {', '.join(blockers[1:])}."
                    if len(blockers) > 1 else ""
                )
                findings.append(_unresolved_claim(
                    claim,
                    "A pairwise comparison cannot bind its subject and comparator to exact recorded structure identifiers."
                    + blocker_text,
                    "IDENTITY_UNRESOLVED",
                    limitations=limitations,
                ))
                continue
            selected = []
            ambiguous = False
            for scheme, molecule in identities:
                matching = [
                    result for result in quantity_results
                    if result.molecule == molecule
                    and (result.molecule_key or "").lower() == scheme
                ]
                if len(matching) != 1:
                    ambiguous = True
                    break
                selected.extend(matching)
            if ambiguous:
                findings.append(_unresolved_claim(
                    claim,
                    "One or more compared identifiers have missing or ambiguous calculation results.",
                    "AMBIGUOUS_RESULT",
                    [_result_evidence(result) for result in selected],
                ))
                continue
            if any(not result.unit for result in selected):
                findings.append(_unresolved_claim(
                    claim,
                    "The pairwise comparison lacks an explicit unit on at least one result.",
                    "MISSING_UNIT",
                    [_result_evidence(result) for result in selected],
                    ["A numeric comparison is not made without explicit units."],
                ))
                continue
            if any(not _is_energy_unit(result.unit) for result in selected):
                findings.append(_unresolved_claim(
                    claim,
                    "At least one comparison unit is not in the supported energy-unit set.",
                    "UNRECOGNIZED_UNIT",
                    [_result_evidence(result) for result in selected],
                ))
                continue
            protocol_issue, missing_fields = _protocol_compatibility(selected)
            if protocol_issue:
                findings.append(_unresolved_claim(
                    claim,
                    f"The pairwise outputs cannot be compared because their calculation protocol is incomplete or incompatible: {protocol_issue}.",
                    protocol_issue,
                    [_result_evidence(result) for result in selected],
                    [
                        "A pairwise scientific comparison requires an explicit compatible calculation protocol.",
                        f"Protocol fields requiring attention: {', '.join(missing_fields) or 'protocol compatibility'}.",
                    ],
                ))
                continue
            if not _same_context(selected):
                findings.append(_unresolved_claim(
                    claim,
                    "The compared outputs do not share one explicit unit and calculation context.",
                    "INCOMPARABLE_CONTEXT",
                    [_result_evidence(result) for result in selected],
                ))
                continue
            subject = selected[0]
            comparisons = [
                (subject.value, comparator.value) for comparator in selected[1:]
            ]
            if any(
                math.isclose(left, right, rel_tol=rel_tol, abs_tol=0.0)
                for left, right in comparisons
            ):
                findings.append(Finding(
                    "claim_evidence", INCONCLUSIVE,
                    "At least one compared pair is tied within the configured relative tolerance.",
                    claim.scope,
                    _claim_evidence(claim) + [_result_evidence(result) for result in selected],
                    ["A near-tie does not establish a strict higher/lower relation."],
                    severity="warning",
                    affected_entities=[result.molecule for result in selected if result.molecule],
                    assessment="indeterminate",
                    reason_code="TIED_RESULTS",
                ))
                continue
            agrees = all(
                (left > right if pairwise_direction == "greater" else left < right)
                for left, right in comparisons
            )
            findings.append(Finding(
                "claim_evidence",
                SUPPORTED if agrees else CONTRADICTED,
                (
                    "The pairwise assertion agrees with all directly identified, comparable trace outputs."
                    if agrees else
                    "The pairwise assertion conflicts with at least one directly identified, comparable trace output."
                ),
                claim.scope,
                _claim_evidence(claim) + [_result_evidence(result) for result in selected],
                [
                    "This is an internal comparison of recorded calculations, not independent scientific validation.",
                    "Only exact structure identifiers are matched; chemical equivalence is not inferred.",
                ],
                severity="warning" if not agrees else "info",
                affected_entities=[result.molecule for result in selected if result.molecule],
            ))
            continue

        if re.search(
            r"\b(?:higher|greater|lower|less)\b.{0,80}\bthan\b.{0,100}"
            r"\b(?:but|whereas|while)\b.{0,30}\b(?:higher|greater|lower|less)\b",
            text,
            re.I,
        ):
            findings.append(_unresolved_claim(
                claim,
                "The sentence combines multiple pairwise directions that this deterministic parser does not separate safely.",
                "MULTIPLE_COMPARISONS",
                limitations=["The statement is retained, but no comparison is selected from a compound sentence."],
            ))
            continue

        direction = _direction(text)
        if direction is None or not re.search(r"\b(?:highest|lowest|maximum|minimum)\b", text, re.I):
            findings.append(_unresolved_claim(
                claim,
                "The assertion is not in one of the supported numeric, pairwise, or explicit-extremum forms.",
                "UNSUPPORTED_ASSERTION_FORM",
            ))
            continue
        if not re.search(r"\b(?:among|of)\b.{0,80}\b(?:tested|computed|candidates?|molecules?)\b", text, re.I):
            findings.append(_unresolved_claim(
                claim,
                "The extremum claim does not define the candidate set over which it applies.",
                "EXTREMUM_SCOPE_UNDEFINED",
            ))
            continue
        findings.append(_unresolved_claim(
            claim,
            "The trace does not establish a complete candidate set or completed search, so a workflow-wide extremum cannot be verified.",
            "CANDIDATE_SET_COMPLETENESS_UNVERIFIED",
            limitations=[
                "A list of recorded results is not evidence that every tested candidate or calculation was captured.",
                "No global maximum or minimum is inferred from a partial provenance trace.",
            ],
        ))
        continue
    return findings


def _ranking_scope_objectives(model: Model, ranking: RankingFact) -> list[Any]:
    return [
        objective for objective in model.objectives
        if objective.scope == ranking.scope
    ]


def _ranking_source_evidence(ranking: RankingFact) -> list[dict[str, Any]]:
    source_refs = ranking.source_refs or [(ranking.record_id, ranking.line_no)]
    return [
        _evidence(
            record_id,
            line_no,
            ranking.field,
            {
                "items": ranking.items,
                "unit": ranking.unit,
                "order_is_source_order": True,
            },
        )
        for record_id, line_no in source_refs
    ]


def _resolve_ranking_identities(
    items: list[dict[str, Any]],
    names: list[str],
    result_pool: list[ResultFact],
) -> tuple[list[Optional[tuple[str, str]]], list[dict[str, Any]]]:
    identities: list[Optional[tuple[str, str]]] = []
    link_evidence: list[dict[str, Any]] = []
    for item, name in zip(items, names):
        scheme = item.get("molecule_key")
        if isinstance(scheme, str) and scheme.lower() in IDENTIFIER_SCHEMES:
            identities.append((scheme.lower(), name))
            continue
        if scheme is not None:
            identities.append(None)
            continue

        candidates = {
            identity
            for result in result_pool
            if result.molecule == name
            and (identity := _result_identity(result)) is not None
        }
        if len(candidates) != 1:
            identities.append(None)
            continue

        identity = next(iter(candidates))
        identities.append(identity)
        matching = [
            result for result in result_pool
            if _result_identity(result) == identity
        ]
        link_evidence.extend(
            _evidence(
                result.record_id,
                result.line_no,
                result.molecule_field or result.field,
                {
                    "exact_value_match_to_typed_identifier": name,
                    "identifier_scheme": identity[0],
                },
            )
            for result in matching
        )
    return identities, link_evidence


def check_ranking_payload_consistency(model: Model) -> list[Finding]:
    findings = []
    results_by_scope = _scope_results(model)
    for ranking in model.rankings:
        result_pool = results_by_scope.get(ranking.scope, [])
        if not result_pool:
            continue

        objective_texts = [
            objective.text
            for objective in _ranking_scope_objectives(model, ranking)
        ]
        quantities = {
            result.quantity for result in result_pool
            if any(_mentions_quantity(text, result.quantity) for text in objective_texts)
        }
        if len(quantities) != 1:
            continue
        quantity = next(iter(quantities))
        raw_names = [item.get("molecule") for item in ranking.items]
        if any(not isinstance(name, str) or not name for name in raw_names):
            continue
        names = [name for name in raw_names if isinstance(name, str)]

        identities, link_evidence = _resolve_ranking_identities(
            ranking.items,
            names,
            result_pool,
        )
        resolved_identities = [
            identity for identity in identities if identity is not None
        ]
        if (
            len(resolved_identities) != len(identities)
            or len(set(resolved_identities)) != len(resolved_identities)
        ):
            continue

        selected_by_identity: dict[tuple[str, str], list[ResultFact]] = defaultdict(list)
        for result in result_pool:
            identity = _result_identity(result)
            if (
                identity is not None
                and identity in resolved_identities
                and norm_text(result.quantity) == norm_text(quantity)
            ):
                selected_by_identity[identity].append(result)
        if any(
            len(selected_by_identity.get(identity, [])) != 1
            for identity in resolved_identities
        ):
            continue
        selected = [
            selected_by_identity[identity][0]
            for identity in resolved_identities
        ]
        stated_values = [item.get("stated_value") for item in ranking.items]
        numeric_values = [
            float(value)
            for value in stated_values
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if (
            len(selected) != len(stated_values)
            or len(numeric_values) != len(stated_values)
            or any(not math.isfinite(value) for value in numeric_values)
        ):
            continue

        agrees = all(
            value == result.value
            for value, result in zip(numeric_values, selected)
        )
        findings.append(Finding(
            "ranking_payload_consistency",
            SUPPORTED if agrees else CONTRADICTED,
            (
                "Ranking numbers exactly match the uniquely linked same-workflow result values."
                if agrees else
                "At least one ranking number differs from its uniquely linked same-workflow result value."
            ),
            ranking.scope,
            _ranking_source_evidence(ranking)
            + link_evidence
            + [_result_evidence(result) for result in selected],
            [
                "This checks numeric payload copying only; it does not establish units, protocol compatibility, ranking direction, or scientific truth."
            ],
            severity="info" if agrees else "warning",
            affected_entities=[name for name in names if name],
            evidence_scope="recorded_payload_consistency_only",
            reason_code="CHECK_PASSED" if agrees else "RANKING_PAYLOAD_VALUE_MISMATCH",
        ))
    return findings


def check_rankings(model: Model, rel_tol: float = 1e-9) -> list[Finding]:
    findings = []
    results_by_scope = _scope_results(model)
    for ranking in model.rankings:
        raw_names = [item.get("molecule") for item in ranking.items]
        evidence = _ranking_source_evidence(ranking)
        if any(not isinstance(name, str) or not name for name in raw_names):
            findings.append(Finding(
                "ranking_consistency",
                INCONCLUSIVE,
                "The ordered list contains missing or duplicate molecule identifiers.",
                ranking.scope,
                evidence,
                ["Array order is preserved as the ranking; no result direction is inferred from numeric values."],
                severity="warning",
                reason_code="MISSING_RANKING_IDENTIFIER",
            ))
            continue
        names: list[str] = [name for name in raw_names if isinstance(name, str)]
        result_pool = results_by_scope.get(ranking.scope, [])
        if not result_pool:
            findings.append(Finding(
                "ranking_consistency",
                INCONCLUSIVE,
                "No calculation outputs occur in this ranking's workflow. Results from other workflows are not matched without an explicit provenance link.",
                ranking.scope,
                evidence,
                [
                    "The snapshot also lacks typed molecule identifiers and an explicit unit.",
                    "The calculation protocol cannot be checked because no same-workflow calculation outputs are present.",
                ],
                severity="warning",
                affected_entities=names,
                reason_code="NO_SCOPED_CALCULATION_RESULTS",
            ))
            continue

        identities, identity_evidence = _resolve_ranking_identities(
            ranking.items,
            names,
            result_pool,
        )
        evidence.extend(identity_evidence)
        if any(identity is None for identity in identities):
            findings.append(Finding(
                "ranking_consistency",
                INCONCLUSIVE,
                "At least one ranking identifier cannot be linked exactly to a typed identifier in a same-workflow calculation.",
                ranking.scope,
                evidence,
                [
                    "Only exact value matches to a unique typed identifier in the same workflow are accepted; chemical names and fuzzy matches are not mapped.",
                    "A missing unit or calculation protocol can still block scientific comparison.",
                ],
                severity="warning",
                affected_entities=names,
                reason_code="IDENTIFIER_SCHEME_MISSING",
            ))
            continue
        identities = [identity for identity in identities if identity is not None]
        if len(set(identities)) != len(identities):
            findings.append(Finding(
                "ranking_consistency",
                INCONCLUSIVE,
                "The ordered list contains duplicate molecule identifiers.",
                ranking.scope,
                evidence,
                ["Duplicate entries prevent a unique comparison."],
                severity="warning",
                reason_code="DUPLICATE_RANKING_IDENTIFIER",
            ))
            continue
        if not ranking.unit:
            findings.append(Finding(
                "ranking_consistency", INCONCLUSIVE,
                "The ranking has no explicit unit, so its values and order cannot be scientifically compared with calculations.",
                ranking.scope,
                evidence,
                ["No unit is inferred from the objective, calculation activity, or numerical magnitude."],
                severity="warning",
                affected_entities=names,
                reason_code="MISSING_RANKING_UNIT",
            ))
            continue
        if not _is_energy_unit(ranking.unit):
            findings.append(Finding(
                "ranking_consistency", INCONCLUSIVE,
                "The ranking unit is not in the supported energy-unit set.",
                ranking.scope,
                evidence,
                ["No unknown unit token is accepted or converted."],
                severity="warning",
                affected_entities=names,
                reason_code="UNRECOGNIZED_RANKING_UNIT",
            ))
            continue
        objectives = _ranking_scope_objectives(model, ranking)
        objective_texts = [objective.text for objective in objectives]
        directions = {direction for text in objective_texts if (direction := _direction(text))}
        result_pool = results_by_scope.get(ranking.scope, [])
        quantities = {
            result.quantity for result in result_pool
            if any(_mentions_quantity(text, result.quantity) for text in objective_texts)
        }
        if len(directions) != 1 or len(quantities) != 1:
            findings.append(Finding(
                "ranking_consistency",
                INCONCLUSIVE,
                "A unique objective direction and scientific quantity cannot be established from recorded objective text.",
                ranking.scope,
                evidence,
                ["Rank order alone does not establish whether larger or smaller values are preferred."],
                severity="warning",
                reason_code="OBJECTIVE_UNRESOLVED",
            ))
            continue
        direction = next(iter(directions))
        quantity = next(iter(quantities))
        selected = [
            result for result in result_pool
            if norm_text(result.quantity) == norm_text(quantity)
            and _result_identity(result) in identities
        ]
        by_identity: dict[tuple[str, str], list[ResultFact]] = defaultdict(list)
        for result in selected:
            identity = _result_identity(result)
            if identity:
                by_identity[identity].append(result)
        if (
            len(by_identity) != len(identities)
            or set(by_identity) != set(identities)
            or any(len(items) != 1 for items in by_identity.values())
            or not selected
        ):
            findings.append(Finding(
                "ranking_consistency",
                INCONCLUSIVE,
                "The ranking does not have exactly one result for every molecule under the same explicit identifier scheme.",
                ranking.scope,
                evidence + [_result_evidence(result) for result in selected],
                ["Chemical names, labels, or bare strings are not equated with typed molecular identifiers."],
                severity="warning",
                affected_entities=[name for name in names if name],
                reason_code="MISSING_OR_AMBIGUOUS_RESULT",
            ))
            continue
        selected = [by_identity[identity][0] for identity in identities]
        if any(not result.unit for result in selected):
            findings.append(Finding(
                "ranking_consistency", INCONCLUSIVE,
                "At least one calculation result has no explicit unit.",
                ranking.scope,
                evidence + [_result_evidence(result) for result in selected],
                ["No unit is assumed or inferred."],
                severity="warning",
                affected_entities=names,
                reason_code="MISSING_RESULT_UNIT",
            ))
            continue
        if any(not _is_energy_unit(result.unit) for result in selected):
            findings.append(Finding(
                "ranking_consistency", INCONCLUSIVE,
                "At least one calculation result has a unit outside the supported energy-unit set.",
                ranking.scope,
                evidence + [_result_evidence(result) for result in selected],
                ["No unknown unit token is accepted or converted."],
                severity="warning",
                affected_entities=names,
                reason_code="UNRECOGNIZED_RESULT_UNIT",
            ))
            continue
        if any(_unit_key(result.unit) != _unit_key(ranking.unit) for result in selected):
            findings.append(Finding(
                "ranking_consistency", INCONCLUSIVE,
                "The ranking and calculation results use different unit codes.",
                ranking.scope,
                evidence + [_result_evidence(result) for result in selected],
                ["No unit conversion is performed; case-sensitive unit symbols are preserved."],
                severity="warning",
                affected_entities=names,
                reason_code="RANKING_UNIT_MISMATCH",
            ))
            continue
        protocol_issue, missing_fields = _protocol_compatibility(selected)
        if protocol_issue:
            findings.append(Finding(
                "ranking_consistency", INCONCLUSIVE,
                f"The ranking cannot be checked against calculations with an incomplete or incompatible protocol: {protocol_issue}.",
                ranking.scope,
                evidence + [_result_evidence(result) for result in selected],
                [
                    "A ranking comparison requires an explicit compatible calculation protocol for every listed molecule.",
                    f"Protocol fields requiring attention: {', '.join(missing_fields) or 'protocol compatibility'}.",
                ],
                severity="warning",
                affected_entities=names,
                reason_code=protocol_issue,
            ))
            continue
        if not _same_context(selected):
            findings.append(Finding(
                "ranking_consistency",
                INCONCLUSIVE,
                "The candidate outputs lack one matching calculation context and explicit common unit.",
                ranking.scope,
                evidence + [_result_evidence(result) for result in selected],
                ["Results from other workflow scopes are not pulled in by campaign ID alone."],
                severity="warning",
                affected_entities=names,
                reason_code="INCOMPARABLE_CONTEXT",
            ))
            continue
        stated_values = [item.get("stated_value") for item in ranking.items]
        has_values = [value is not None for value in stated_values]
        if any(has_values) and not all(has_values):
            findings.append(Finding(
                "ranking_consistency", INCONCLUSIVE,
                "Only some ranking items carry numeric values, so the snapshot cannot be fully cross-checked.",
                ranking.scope,
                evidence + [_result_evidence(result) for result in selected],
                ["Partial ranking values are not silently ignored."],
                severity="warning",
                affected_entities=names,
                reason_code="PARTIAL_RANKING_VALUES",
            ))
            continue
        if all(has_values):
            numeric_values = [
                float(value)
                for value in stated_values
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
            if (
                len(numeric_values) != len(stated_values)
                or any(not math.isfinite(value) for value in numeric_values)
            ):
                findings.append(Finding(
                    "ranking_consistency", INCONCLUSIVE,
                    "The ranking contains a non-numeric or non-finite value.",
                    ranking.scope,
                    evidence + [_result_evidence(result) for result in selected],
                    severity="warning",
                    affected_entities=names,
                    reason_code="NONFINITE_RANKING_VALUE",
                ))
                continue
            values_agree = all(
                math.isclose(item_value, result.value, rel_tol=rel_tol, abs_tol=0.0)
                for item_value, result in zip(numeric_values, selected)
            )
            if not values_agree:
                findings.append(Finding(
                    "ranking_consistency", CONTRADICTED,
                    "At least one stated ranking value conflicts with the uniquely matched calculation output.",
                    ranking.scope,
                    evidence + [_result_evidence(result) for result in selected],
                    ["This detects a trace-level value mismatch; it does not establish scientific truth."],
                    severity="warning",
                    affected_entities=names,
                    reason_code="RANKING_VALUE_MISMATCH",
                ))
                continue
        values = [result.value for result in selected]
        if any(
            math.isclose(values[index], values[index + 1], rel_tol=rel_tol, abs_tol=0.0)
            for index in range(len(values) - 1)
        ):
            findings.append(Finding(
                "ranking_consistency", INCONCLUSIVE,
                "Adjacent ranked candidates are tied within the configured relative tolerance.",
                ranking.scope,
                evidence + [_result_evidence(result) for result in selected],
                ["A tied pair does not establish a strict relative rank."],
                severity="warning",
                affected_entities=names,
                assessment="indeterminate",
                reason_code="TIED_RESULTS",
            ))
            continue
        ordered = all(
            values[index] >= values[index + 1] if direction == "max"
            else values[index] <= values[index + 1]
            for index in range(len(values) - 1)
        )
        findings.append(Finding(
            "ranking_consistency",
            SUPPORTED if ordered else CONTRADICTED,
            (
                "The source order and all stated numeric values agree within the listed ranking subset and comparable recorded results."
                if ordered and all(has_values) else
                "The source order matches the explicit objective within the listed ranking subset."
                if ordered else
                "The source order conflicts with the explicit objective within the listed ranking subset."
            ),
            ranking.scope,
            evidence + [_result_evidence(result) for result in selected],
            [
                "This verifies trace-level consistency, not scientific truth or experimental validity.",
                "The snapshot is checked only as a listed subset; it does not establish that the broader candidate set is complete or that the top item is a campaign-wide optimum.",
            ],
            severity="warning" if not ordered else "info",
            affected_entities=names,
            reason_code="CHECK_PASSED" if ordered else "RANKING_ORDER_MISMATCH",
        ))
    return findings


def check_provenance(model: Model) -> list[Finding]:
    findings = []
    graph = model.graph
    for record in model.dataset.records:
        if record.issues:
            findings.append(Finding(
                "record_structure",
                INCONCLUSIVE,
                "The record has incomplete or uncertain provenance metadata.",
                record.scope_label,
                [_evidence(record.uid, record.line_no, "record_metadata", record.issues)],
                ["Missing metadata is not evidence that a scientific result is false."],
                severity="warning",
            ))
        for parent_ref in record.parents:
            parent, method = resolve_record_reference(record, parent_ref, model.dataset)
            if not parent:
                ambiguous = method.startswith("ambiguous")
                findings.append(Finding(
                    "parent_reference",
                    INCONCLUSIVE,
                    (
                        f"The parent ID matches multiple records in the export and cannot be resolved safely ({method})."
                        if ambiguous else
                        "The parent ID was not found among normalized record IDs in this export; it may refer to a record outside this export."
                    ),
                    record.scope_label,
                    _parent_reference_evidence(record, parent_ref),
                    [
                        "Not found in this export does not establish that the reference is invalid."
                        if not ambiguous else
                        "No parent relationship is guessed from record order or similar content.",
                        "Confirm the parent is included and its ID is stored in a recognized identifier field."
                        if not ambiguous else
                        "Provide unique record IDs or an explicit disambiguating reference."
                    ],
                    severity="warning",
                    reason_code=(
                        "PARENT_REFERENCE_AMBIGUOUS"
                        if ambiguous else "PARENT_ID_NOT_FOUND_IN_EXPORT"
                    ),
                ))
                continue
            cross_scope = parent.scope != record.scope
            relation_edges = graph.get_edge_data(f"evidence::{record.uid}", f"evidence::{parent.uid}", default={})
            linked = any(edge.get("relation") == "derived_from" for edge in relation_edges.values())
            findings.append(Finding(
                "parent_reference",
                SUPPORTED if linked else INCONCLUSIVE,
                (
                    "The explicit parent reference resolves uniquely and is represented in the evidence graph."
                    if linked else
                    "The reference resolves, but a unique provenance edge was not constructed."
                ),
                record.scope_label,
                [
                    *_parent_reference_evidence(record, parent_ref),
                    _evidence(parent.uid, parent.line_no, "original_id", parent.original_id),
                ],
                ["Parent linkage records lineage; it does not by itself validate scientific content."],
                affected_entities=[record.uid, parent.uid],
            ))
            if cross_scope:
                findings[-1].limitations.append(
                    "The explicit parent crosses workflow scopes; its semantics depend on the source system."
                )

    edges = [
        (source, target)
        for source, target, attrs in graph.edges(data=True)
        if attrs.get("relation") == "derived_from"
    ]
    adjacency: dict[str, list[str]] = defaultdict(list)
    for source, target in edges:
        adjacency[source].append(target)
    cycle = _find_cycle(adjacency)
    if cycle:
        by_uid = {f"evidence::{record.uid}": record for record in model.dataset.records}
        cycle_evidence = []
        for child_node, parent_node in zip(cycle, cycle[1:]):
            child = by_uid.get(child_node)
            parent = by_uid.get(parent_node)
            if not child or not parent:
                continue
            for reference in child.parents:
                resolved, _ = resolve_record_reference(child, reference, model.dataset)
                if resolved and resolved.uid == parent.uid:
                    cycle_evidence.extend(
                        _parent_reference_evidence(child, reference)
                    )
        findings.append(Finding(
            "provenance_cycle",
            CONTRADICTED,
            "Explicitly resolved parent relationships contain a directed cycle.",
            None,
            cycle_evidence,
            ["The cycle is based only on uniquely resolved explicit references."],
            severity="error",
            affected_entities=cycle,
        ))
    for record in model.dataset.records:
        if record.parent_workflow_id:
            parent_workflows = [
                item for item in model.dataset.workflow_records
                if item.workflow_id == record.parent_workflow_id
                and item.campaign_id == record.campaign_id
            ]
            if len(parent_workflows) != 1:
                missing = not parent_workflows
                findings.append(Finding(
                    "parent_workflow_reference",
                    INCONCLUSIVE,
                    (
                        "The parent workflow ID did not match a workflow record in this campaign."
                        if missing else
                        "The parent workflow ID matches multiple records in this campaign."
                    ),
                    record.scope_label,
                    [_evidence(record.uid, record.line_no, "parent_workflow_id", record.parent_workflow_id)],
                    severity="warning",
                    reason_code=(
                        "PARENT_WORKFLOW_NOT_IN_CAMPAIGN"
                        if missing else "PARENT_WORKFLOW_AMBIGUOUS"
                    ),
                ))
            else:
                parent = parent_workflows[0]
                findings.append(Finding(
                    "parent_workflow_reference",
                    SUPPORTED,
                    "The explicit parent workflow resolves uniquely in the same campaign.",
                    record.scope_label,
                    [
                        _evidence(record.uid, record.line_no, "parent_workflow_id", record.parent_workflow_id),
                        _evidence(parent.uid, parent.line_no, "workflow_id", parent.workflow_id),
                    ],
                ))
    return findings


def _find_cycle(adjacency: dict[str, list[str]]) -> list[str]:
    visited: set[str] = set()
    active: list[str] = []
    active_set: set[str] = set()

    def visit(node: str) -> Optional[list[str]]:
        if node in active_set:
            start = active.index(node)
            return active[start:] + [node]
        if node in visited:
            return None
        visited.add(node)
        active.append(node)
        active_set.add(node)
        for neighbor in adjacency.get(node, []):
            result = visit(neighbor)
            if result:
                return result
        active.pop()
        active_set.remove(node)
        return None

    for node in adjacency:
        result = visit(node)
        if result:
            return result
    return []


def _parent_reference_evidence(
    record: Any,
    reference: str,
) -> list[dict[str, Any]]:
    source_fields = list(dict.fromkeys(
        field_name
        for field_name, parent_id in record.parent_references
        if parent_id == reference
    )) or ["parents"]
    return [
        _evidence(record.uid, record.line_no, field_name, reference)
        for field_name in source_fields
    ]


def check_missing_evidence(model: Model) -> list[Finding]:
    findings = []
    for calc in model.calcs:
        if not calc.results and not calc.nonfinite:
            findings.append(Finding(
                "missing_result_evidence",
                INCONCLUSIVE,
                "A candidate scientific calculation has no extractable finite numerical output.",
                calc.scope,
                [_evidence(calc.record.uid, calc.record.line_no, "generated", {
                    "status": calc.record.status,
                    "molecule": calc.molecule,
                    "identity_conflict": calc.identity_conflict,
                })],
                ["The calculation classifier is heuristic; inspect source fields if the output schema is new."],
                severity="warning",
                affected_entities=[calc.molecule] if calc.molecule else [],
            ))
        for path, value in calc.nonfinite:
            findings.append(Finding(
                "nonfinite_result",
                INCONCLUSIVE,
                "A generated numerical field is non-finite and cannot be treated as a scientific result.",
                calc.scope,
                [_evidence(calc.record.uid, calc.record.line_no, path, repr(value))],
                ["Non-finite values are excluded from comparisons."],
                severity="warning",
            ))
    for error in model.dataset.parse_errors:
        findings.append(Finding(
            "input_parse_error",
            INCONCLUSIVE,
            "An input JSONL line could not be parsed.",
            None,
            [_evidence(None, error.get("line_no"), "jsonl", error.get("error"))],
            ["Malformed lines are reported and excluded from extraction."],
            severity="warning",
        ))
    return findings


def verify_all(model: Model, rel_tol: float = 1e-9) -> list[Finding]:
    if not math.isfinite(rel_tol) or rel_tol < 0:
        raise ValueError("rel_tol must be a finite non-negative number")
    return (
        check_provenance(model)
        + check_association(model)
        + check_calculation_protocol(model)
        + check_numeric_consistency(model, rel_tol)
        + check_claims(model, rel_tol)
        + check_ranking_payload_consistency(model)
        + check_rankings(model, rel_tol)
        + check_missing_evidence(model)
    )


def summarize(findings: list[Finding]) -> dict[str, int]:
    return {
        status: sum(finding.status == status for finding in findings)
        for status in (SUPPORTED, CONTRADICTED, INCONCLUSIVE)
    }


def summarize_scientific_checks(findings: list[Finding]) -> dict[str, int]:
    return summarize([
        finding for finding in findings
        if finding.check in SCIENTIFIC_ASSERTION_CHECKS
    ])


def summarize_by_check(findings: list[Finding]) -> dict[str, dict[str, int]]:
    checks = sorted({finding.check for finding in findings})
    return {
        check: summarize([finding for finding in findings if finding.check == check])
        for check in checks
    }


def summarize_assessments(findings: list[Finding]) -> dict[str, int]:
    return {
        assessment: sum(finding.assessment == assessment for finding in findings)
        for assessment in ("performed", "not_performed", "indeterminate")
    }


def write_reports(
    model: Model,
    findings: list[Finding],
    output_dir: str | Path,
    input_path: str,
    rel_tol: float,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    counts = summarize(findings)
    protocol_issues = defaultdict(int)
    for result in model.results:
        issue, _ = _protocol_issue(result)
        protocol_issues[issue or "complete"] += 1
    graph = graph_to_dict(model.graph)
    results_by_scope = _scope_results(model)
    rankings_with_scoped_results = sum(
        bool(results_by_scope.get(ranking.scope))
        for ranking in model.rankings
    )
    findings_by_reason: dict[tuple[str, str, str], list[Finding]] = defaultdict(list)
    for finding in findings:
        findings_by_reason[
            (finding.check, finding.status, finding.reason_code or "")
        ].append(finding)
    parent_reference_audit: dict[str, list[dict[str, Any]]] = {
        "unresolved": [],
        "ambiguous": [],
    }
    parent_audit_codes = {
        "PARENT_ID_NOT_FOUND_IN_EXPORT": "unresolved",
        "PARENT_REFERENCE_AMBIGUOUS": "ambiguous",
    }
    for finding in findings:
        audit_group = parent_audit_codes.get(finding.reason_code or "")
        if not audit_group:
            continue
        for item in finding.evidence:
            reference = item.get("detail")
            if not isinstance(reference, str):
                continue
            ref_entry = {
                "source_record_id": item.get("record_id"),
                "line_no": item.get("line_no"),
                "source_field": item.get("field"),
                "referenced_id": reference,
            }
            if ref_entry not in parent_reference_audit[audit_group]:
                parent_reference_audit[audit_group].append(ref_entry)
    payload = {
        "project": "Provenance Analytics for Trustworthy AI Systems",
        "input": input_path,
        "input_sha256": model.dataset.sha256,
        "total_lines": model.dataset.total_lines,
        "parsed_records": len(model.dataset.records),
        "parse_errors": model.dataset.parse_errors,
        "relative_tolerance": rel_tol,
        "summary": counts,
        "assessment_summary": summarize_assessments(findings),
        "trace_consistency_checks": summarize_scientific_checks(findings),
        "recorded_payload_consistency": summarize([
            finding for finding in findings
            if finding.check == "ranking_payload_consistency"
        ]),
        "independent_scientific_validation": {
            "performed": False,
            "reason": (
                "The offline system checks recorded trace consistency only; it has no independent "
                "scientific evidence source or validated calculation backend."
            ),
        },
        "checks_by_type": summarize_by_check(findings),
        "parent_reference_audit": {
            "unresolved_count": len(parent_reference_audit["unresolved"]),
            "unresolved_distinct_id_count": len({
                item["referenced_id"]
                for item in parent_reference_audit["unresolved"]
            }),
            "unresolved": parent_reference_audit["unresolved"],
            "ambiguous_count": len(parent_reference_audit["ambiguous"]),
            "ambiguous": parent_reference_audit["ambiguous"],
        },
        "coverage": {
            "campaigns": len({record.campaign_id for record in model.dataset.records if record.campaign_id}),
            "workflow_scopes": len(model.scopes),
            "candidate_calculations": len(model.calcs),
            "finite_results": len(model.results),
            "results_with_units": sum(bool(result.unit) for result in model.results),
            "results_without_units": sum(not result.unit for result in model.results),
            "results_with_complete_protocol": protocol_issues["complete"],
            "results_with_incomplete_protocol": len(model.results) - protocol_issues["complete"],
            "protocol_issue_counts": dict(sorted(
                (issue, count)
                for issue, count in protocol_issues.items()
                if issue != "complete"
            )),
            "claim_source_fields": model.claim_source_fields,
            "claims": len(model.claims),
            "claim_text_outside_supported_forms": model.claim_text_outside_supported_forms,
            "objectives": len(model.objectives),
            "rankings": len(model.rankings),
            "rankings_with_same_workflow_results": rankings_with_scoped_results,
            "rankings_without_same_workflow_results": (
                len(model.rankings) - rankings_with_scoped_results
            ),
        },
        "claim_assessment_scope": (
            "Deterministic checks compare explicit numeric IE assertions, direct pairwise "
            "comparisons, and explicitly scoped ranking subsets against local recorded outputs. "
            "These are trace-consistency checks, not independent scientific validation. Other "
            "narrative hypotheses, mechanistic explanations, and recommendations are not verified. "
            "Workflow-wide extrema remain inconclusive because this trace has no completeness marker."
        ),
        "findings": [finding.to_dict() for finding in findings],
        "knowledge_graph": graph,
        "limitations": [
            "Extraction uses conservative field and activity-name heuristics.",
            "SUPPORTED means the stated check agrees with recorded trace evidence; it does not establish scientific truth.",
            "Missing units, ambiguous identity, or incomplete comparison context lead to INCONCLUSIVE findings.",
            "Numeric scientific comparisons also require an explicit scientific_protocol object with method, software/version, ionization type, neutral/cation charge and multiplicity, geometry protocol, convergence, energy unit, and method parameters.",
            "A complete protocol object establishes metadata completeness only; it does not establish that the method is scientifically accurate.",
            "Chemical names are not mapped to structure identifiers; only typed exact identifiers are linked.",
            "Unit matching is case-sensitive, no conversions are performed, and only a limited energy-unit allowlist is accepted.",
            "Narrative hypotheses and mechanistic explanations outside supported assertion forms are not verified.",
            "No external scientific evidence or online retrieval is used.",
        ],
    }
    json_path = output / "verification_report.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    coverage = payload["coverage"]
    lines = [
        "# Provenance Verification Summary",
        "",
        "**Scientific conclusion: no claim was independently validated.** "
        "This offline run checks recorded trace consistency only.",
        "",
        f"- Input: `{input_path}`",
        f"- SHA-256: `{model.dataset.sha256 or 'not available'}`",
        f"- Records: {len(model.dataset.records)} parsed / {model.dataset.total_lines} lines; "
        f"{len(model.dataset.parse_errors)} parse errors",
        "",
        "## Evidence coverage",
        "",
        "| Checkpoint | Result |",
        "|---|---:|",
        f"| Calculation outputs | {coverage['finite_results']} finite |",
        f"| Outputs with explicit units | {coverage['results_with_units']} / {coverage['finite_results']} |",
        f"| Outputs with complete protocols | {coverage['results_with_complete_protocol']} / {coverage['finite_results']} |",
        f"| Rankings with same-workflow calculation outputs | {coverage['rankings_with_same_workflow_results']} / {coverage['rankings']} |",
        f"| Extracted claims / rankings | {coverage['claims']} / {coverage['rankings']} |",
        f"| Narrative text outside supported forms | {coverage['claim_text_outside_supported_forms']} (not assessed) |",
        "",
        "## Parent-reference audit",
        "",
        (
            f"Not found in this export: {payload['parent_reference_audit']['unresolved_count']} "
            f"references across {payload['parent_reference_audit']['unresolved_distinct_id_count']} IDs; "
            f"ambiguous: {payload['parent_reference_audit']['ambiguous_count']}. "
            "Not found does not mean invalid; these references remain INCONCLUSIVE."
        ),
        "",
        "| Resolution | Source record | Line | Source field | Referenced ID |",
        "|---|---|---:|---|---|",
    ]
    for group, label in (
        ("unresolved", "Not found in export"),
        ("ambiguous", "Ambiguous in export"),
    ):
        for item in payload["parent_reference_audit"][group]:
            lines.append(
                f"| {label} | `{item['source_record_id']}` | {item['line_no']} | "
                f"`{item['source_field']}` | `{item['referenced_id']}` |"
            )
    if not any(payload["parent_reference_audit"][key] for key in ("unresolved", "ambiguous")):
        lines.append("| None | - | - | - | - |")
    lines.extend([
        "",
        "## Results",
        "",
        "| Result scope | SUPPORTED | CONTRADICTED | INCONCLUSIVE |",
        "|---|---:|---:|---:|",
    ])
    structural_findings = [
        finding for finding in findings
        if finding.check not in SCIENTIFIC_ASSERTION_CHECKS
        and finding.check != "ranking_payload_consistency"
    ]
    structural_counts = summarize(structural_findings)
    lines.append(
        f"| Provenance and metadata checks | {structural_counts[SUPPORTED]} | "
        f"{structural_counts[CONTRADICTED]} | {structural_counts[INCONCLUSIVE]} |"
    )
    trace_counts = payload["trace_consistency_checks"]
    lines.append(
        f"| Scientific trace comparisons | {trace_counts[SUPPORTED]} | "
        f"{trace_counts[CONTRADICTED]} | {trace_counts[INCONCLUSIVE]} |"
    )
    payload_counts = payload["recorded_payload_consistency"]
    lines.append(
        f"| Raw ranking-value copies | {payload_counts[SUPPORTED]} | "
        f"{payload_counts[CONTRADICTED]} | {payload_counts[INCONCLUSIVE]} |"
    )
    lines.extend([
        "",
        "SUPPORTED provenance findings mean only that a recorded link or association exists. "
        "They do not validate the chemical structure, calculation, units, or scientific claim.",
        "",
        "## Findings by issue",
        "",
        "| Check | Status | Reason | Count | Example source records |",
        "|---|---|---|---:|---|",
    ])
    for (check, status, reason), grouped in sorted(findings_by_reason.items()):
        record_ids = list(dict.fromkeys(
            record_id
            for finding in grouped
            for record_id in finding.source_record_ids
        ))
        examples = ", ".join(f"`{record_id}`" for record_id in record_ids[:3]) or "—"
        if len(record_ids) > 3:
            examples += f", +{len(record_ids) - 3} more"
        lines.append(
            f"| {check} | {status} | `{reason}` | {len(grouped)} | {examples} |"
        )
    lines.extend([
        "",
        "## What blocks stronger conclusions",
        "",
        "- The 16 calculation outputs have no recorded units or complete scientific protocols.",
        "- Ranking records are in workflows with no calculation outputs; cross-workflow matches "
        "are not made without explicit provenance links.",
        "- Narrative chemical names are not mapped to structures, and the trace has no complete-search marker.",
        "- No independent evidence source or validated calculation backend is available.",
        "",
        "For every finding's full message, limitations, and record/line/field evidence, see "
        "`verification_report.json`. The graph is in `knowledge_graph.json`.",
        "",
        "## Recommended next steps",
        "",
        "1. Emit explicit units, method/software versions, charge/spin, geometry protocol, "
        "convergence, and calculation parameters with each result.",
        "2. Give ranking items typed molecule identifiers and link ranking workflows to their "
        "calculation workflows using explicit parent IDs.",
        "3. Record expected candidates and completion/failure states before making global-extremum claims.",
        "4. Add independent evidence retrieval only as a separate, source-cited verification layer.",
    ])
    markdown_path = output / "verification_report.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    graph_path = output / "knowledge_graph.json"
    graph_path.write_text(
        json.dumps(graph, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return {
        "json": str(json_path),
        "markdown": str(markdown_path),
        "graph": str(graph_path),
    }
