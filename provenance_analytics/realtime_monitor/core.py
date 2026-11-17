"""Stable adapter to the independent batch parsing/extraction/verifier core."""
from __future__ import annotations

from collections.abc import Iterable
import re
from typing import Any

from knowledge_graph import extract
from provenance import from_dicts
from verification import CONTRADICTED, INCONCLUSIVE, SUPPORTED, Finding, verify_all


def analyze_record_snapshot(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run the established offline verifier over one current run snapshot."""
    records = list(records)
    model = extract(from_dicts(records))
    findings = [finding.to_dict() for finding in verify_all(model)]
    # Surface failed GAIED candidate calculations explicitly. The batch verifier
    # intentionally avoids interpreting application stderr, while the live loop
    # needs a grounded, actionable finding for a recorded RDKit parse failure.
    known_ids = {record.original_id: record for record in model.dataset.records}
    raw_by_id = {
        str(item.get("task_id") or item.get("activity_id") or item.get("id") or ""): item
        for item in records
    }
    for record_id, raw in zip(
        (str(item.get("task_id") or item.get("activity_id") or item.get("id") or "") for item in records),
        records,
    ):
        activity = str(raw.get("activity_id", ""))
        status = str(raw.get("status", "")).upper()
        stderr = str(raw.get("stderr") or "")
        used = raw.get("used") if isinstance(raw.get("used"), dict) else {}
        smiles = used.get("smiles")
        if activity != "compute_ionization_energy" or not smiles:
            continue
        graph_record = known_ids.get(record_id)
        if graph_record is None:
            continue
        workflow = raw.get("workflow_id")
        campaign = raw.get("campaign_id")
        scope = f"campaign-{campaign}/workflow-{workflow}"
        generated = raw.get("generated") if isinstance(raw.get("generated"), dict) else {}
        if status in {"ERROR", "FAILED"} and "parse failure" in stderr.casefold():
            finding = Finding(
                "candidate_input_validity", CONTRADICTED,
                f"GAIED could not parse candidate SMILES {smiles!r}; xTB did not produce an energy.",
                scope,
                [{"record_id": graph_record.uid, "line_no": graph_record.line_no,
                  "field": "stderr", "detail": stderr},
                 {"record_id": graph_record.uid, "line_no": graph_record.line_no,
                  "field": "used.smiles", "detail": str(smiles)}],
                ["The finding reports the recorded RDKit parse failure; it does not infer a corrected structure."],
                severity="error", affected_entities=[str(smiles)],
                reason_code="INPUT_PARSE_ERROR",
                evidence_scope="recorded_execution_failure",
            )
            findings.append(finding.to_dict())
        elif status in {"FINISHED", "SUCCEEDED", "SUCCESS"} and generated.get("ionization_energy") is not None:
            finding = Finding(
                "candidate_input_validity", SUPPORTED,
                f"GAIED parsed and completed xTB ionization-energy calculation for {smiles!r}.",
                scope,
                [{"record_id": graph_record.uid, "line_no": graph_record.line_no,
                  "field": "generated.ionization_energy", "detail": generated.get("ionization_energy")},
                 {"record_id": graph_record.uid, "line_no": graph_record.line_no,
                  "field": "used.smiles", "detail": str(smiles)}],
                ["Execution success confirms the workflow accepted this structure; it does not independently validate scientific accuracy."],
                severity="info", affected_entities=[str(smiles)],
                reason_code="CHECK_PASSED",
                evidence_scope="recorded_execution",
            )
            findings.append(finding.to_dict())

    # Compare GAIED's latest natural-language conclusion with the exact numeric
    # molecule map it records. This catches trace contradictions without
    # pretending that matching the trace validates the chemistry itself.
    failing_phrases = (
        "invalid smiles", "invalid smi", "parse failure", "could not be run",
        "could not run", "simulation failed", "failed to calculate", "-inf",
    )
    for raw in records:
        if raw.get("activity_id") != "langgraph/conclude":
            continue
        generated = raw.get("generated") if isinstance(raw.get("generated"), dict) else {}
        values = generated.get("simulated_molecules")
        conclusions = generated.get("conclusions")
        if not isinstance(values, dict) or not isinstance(conclusions, list):
            continue
        conclusion_id = str(raw.get("task_id") or raw.get("activity_id") or raw.get("id") or "")
        conclusion_record = known_ids.get(conclusion_id)
        if conclusion_record is None:
            continue
        campaign = raw.get("campaign_id")
        workflow = raw.get("workflow_id")
        scope = f"campaign-{campaign}/workflow-{workflow}"
        texts = [str(item) for item in conclusions if isinstance(item, str)]
        latest = texts[-1] if texts else ""
        for smiles, value in values.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            matching = [text for text in texts if str(smiles).casefold() in text.casefold()]
            latest_matching = matching[-1] if matching else ""
            latest_is_failure = any(
                phrase in latest_matching.casefold() for phrase in failing_phrases
            )
            calc_record = next((
                graph_record for source_id, graph_record in known_ids.items()
                if (record := raw_by_id.get(source_id, {})).get("activity_id") == "compute_ionization_energy"
                and isinstance(record.get("used"), dict)
                and record["used"].get("smiles") == smiles
                and isinstance(record.get("generated"), dict)
                and record["generated"].get("ionization_energy") == value
            ), None)
            if calc_record is None:
                continue
            calc_raw = next((
                record for source_id, record in raw_by_id.items()
                if known_ids.get(source_id) is calc_record
            ), {})
            calc_generated = calc_raw.get("generated") if isinstance(calc_raw.get("generated"), dict) else {}
            calc_metadata = calc_raw.get("custom_metadata") if isinstance(calc_raw.get("custom_metadata"), dict) else {}
            recorded_unit = calc_generated.get("unit") or calc_metadata.get("unit")
            if not isinstance(recorded_unit, str):
                recorded_unit = None
            bad_records = []
            unit_bad_records = []
            for source_id, source in raw_by_id.items():
                source_generated = source.get("generated") if isinstance(source.get("generated"), dict) else {}
                source_claims = source_generated.get("conclusions")
                if source.get("activity_id") != "langgraph/conclude" or not isinstance(source_claims, list):
                    continue
                related_claims = [
                    claim for claim in source_claims
                    if isinstance(claim, str)
                    and str(smiles).casefold() in claim.casefold()
                ]
                if related_claims and any(
                    phrase in related_claims[-1].casefold()
                    for phrase in failing_phrases
                ):
                    graph_source = known_ids.get(source_id)
                    if graph_source is not None:
                        bad_records.append(graph_source)
                if related_claims and recorded_unit:
                    prior_unit = re.search(
                        r"\b(e\s*v|electron\s*volts?|kj\s*/\s*mol|kj\s*mol(?:\s*\^?\s*-?1)?|hartrees?|ha)\b",
                        related_claims[-1], flags=re.IGNORECASE,
                    )
                    if prior_unit:
                        prior = re.sub(r"\s+", "", prior_unit.group(1)).casefold()
                        expected = re.sub(r"\s+", "", recorded_unit).casefold()
                        aliases = {"electronvolt": "ev", "electronvolts": "ev", "hartree": "ha", "hartrees": "ha"}
                        if aliases.get(prior, prior) != aliases.get(expected, expected):
                            graph_source = known_ids.get(source_id)
                            if graph_source is not None:
                                unit_bad_records.append(graph_source)
            unit_match = None
            unit_mismatch = False
            if latest_matching and recorded_unit:
                unit_match = re.search(
                    r"\b(e\s*v|electron\s*volts?|kj\s*/\s*mol|kj\s*mol(?:\s*\^?\s*-?1)?|hartrees?|ha)\b",
                    latest_matching,
                    flags=re.IGNORECASE,
                )
                if unit_match:
                    claimed_unit = re.sub(r"\s+", "", unit_match.group(1)).casefold()
                    expected_unit = re.sub(r"\s+", "", recorded_unit).casefold()
                    aliases = {"electronvolt": "ev", "electronvolts": "ev", "hartree": "ha", "hartrees": "ha"}
                    unit_mismatch = aliases.get(claimed_unit, claimed_unit) != aliases.get(expected_unit, expected_unit)
            if latest_is_failure:
                unit_suffix = f" {recorded_unit}" if recorded_unit else ""
                finding = Finding(
                    "claim_evidence", CONTRADICTED,
                    f"GAIED's conclusion says {smiles!r} failed or was invalid, but the same-workflow calculation record contains a finite ionization energy ({value}{unit_suffix}).",
                    scope,
                    [{"record_id": conclusion_record.uid, "line_no": conclusion_record.line_no,
                      "field": "generated.conclusions", "detail": latest_matching},
                     {"record_id": calc_record.uid, "line_no": calc_record.line_no,
                      "field": "generated.ionization_energy", "detail": value}],
                    ["This detects disagreement with recorded output; a finite trace value is not independent proof of scientific accuracy."],
                    severity="error", affected_entities=[str(smiles)],
                    reason_code="CHECK_CONFLICT", evidence_scope="recorded_trace_consistency",
                )
                findings.append(finding.to_dict())
            elif unit_mismatch:
                finding = Finding(
                    "claim_evidence", CONTRADICTED,
                    f"GAIED labels the {smiles!r} ionization-energy result as {unit_match.group(1)}, but the calculation record specifies {recorded_unit}.",
                    scope,
                    [{"record_id": conclusion_record.uid, "line_no": conclusion_record.line_no,
                      "field": "generated.conclusions", "detail": latest_matching},
                     {"record_id": calc_record.uid, "line_no": calc_record.line_no,
                      "field": "generated.unit", "detail": recorded_unit},
                     {"record_id": calc_record.uid, "line_no": calc_record.line_no,
                      "field": "generated.ionization_energy", "detail": value}],
                    ["This checks consistency with the recorded unit; it does not independently validate the calculation."],
                    severity="error", affected_entities=[str(smiles)],
                    reason_code="UNIT_MISMATCH", evidence_scope="recorded_trace_consistency",
                )
                findings.append(finding.to_dict())
            elif unit_bad_records and unit_match and str(smiles).casefold() in latest.casefold() and any(
                phrase in latest.casefold() for phrase in ("energy", "result", "computed", "calculated")
            ):
                evidence = [
                    {"record_id": item.uid, "line_no": item.line_no,
                     "field": "generated.conclusions", "detail": "Prior conclusion used a unit that conflicts with the calculation record."}
                    for item in unit_bad_records
                ]
                evidence.extend([
                    {"record_id": calc_record.uid, "line_no": calc_record.line_no,
                     "field": "generated.unit", "detail": recorded_unit},
                    {"record_id": calc_record.uid, "line_no": calc_record.line_no,
                     "field": "generated.ionization_energy", "detail": value},
                    {"record_id": conclusion_record.uid, "line_no": conclusion_record.line_no,
                     "field": "generated.conclusions", "detail": latest},
                ])
                findings.append(Finding(
                    "claim_evidence", SUPPORTED,
                    f"A later GAIED conclusion uses the recorded unit {recorded_unit} for {smiles!r}.",
                    scope, evidence,
                    ["This verifies unit consistency with the trace, not scientific truth."],
                    severity="info", affected_entities=[str(smiles)],
                    reason_code="CHECK_PASSED", evidence_scope="recorded_trace_consistency",
                ).to_dict())
            elif bad_records and str(smiles).casefold() in latest.casefold() and any(
                phrase in latest.casefold() for phrase in ("energy", "result", "computed", "calculated", "ha")
            ):
                evidence = [
                    {"record_id": item.uid, "line_no": item.line_no,
                     "field": "generated.conclusions", "detail": "Prior conflicting conclusion addressed by this later conclusion."}
                    for item in bad_records
                ]
                evidence.extend([
                    {"record_id": calc_record.uid, "line_no": calc_record.line_no,
                     "field": "generated.ionization_energy", "detail": value},
                    {"record_id": conclusion_record.uid, "line_no": conclusion_record.line_no,
                     "field": "generated.conclusions", "detail": latest},
                ])
                finding = Finding(
                    "claim_evidence", SUPPORTED,
                    f"A later GAIED conclusion addresses {smiles!r} using the recorded finite xTB result.",
                    scope, evidence,
                    ["This verifies consistency with the trace, not scientific truth."],
                    severity="info", affected_entities=[str(smiles)],
                    reason_code="CHECK_PASSED", evidence_scope="recorded_trace_consistency",
                )
                findings.append(finding.to_dict())
    return findings
