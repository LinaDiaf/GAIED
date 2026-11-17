"""Focused tests for conservative provenance extraction and verification."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge_graph import extract  # noqa: E402
from provenance import from_dicts, parse_lines  # noqa: E402
from verification import (  # noqa: E402
    CONTRADICTED,
    INCONCLUSIVE,
    SUPPORTED,
    verify_all,
    write_reports,
)


def calc(
    task_id,
    smiles,
    value,
    *,
    workflow="wf1",
    campaign="c1",
    status="FINISHED",
    unit=None,
    basis="xtb",
    parent_task_id=None,
    with_protocol=True,
):
    output = {"ionization_energy": value}
    if unit is not None:
        output["unit"] = unit
    record = {
        "task_id": task_id,
        "workflow_id": workflow,
        "campaign_id": campaign,
        "activity_id": "compute_ionization_energy",
        "subtype": "tool_call",
        "status": status,
        "custom_metadata": {"tool_name": "xtb"},
        "used": {"smiles": smiles, "basis": basis},
        "generated": output,
    }
    if with_protocol:
        record["custom_metadata"]["scientific_protocol"] = {
            "method": "GFN2-xTB",
            "software": "xtb-python",
            "software_version": "test-version",
            "ionization_energy_type": "vertical",
            "geometry_protocol": "neutral_optimized_cation_single_point",
            "neutral_charge": 0,
            "cation_charge": 1,
            "neutral_multiplicity": 1,
            "cation_multiplicity": 2,
            "all_geometry_optimizations_converged": True,
            "energy_unit": unit or "eV",
            "method_parameters": {"basis": basis},
        }
    if parent_task_id:
        record["parent_task_id"] = parent_task_id
    return record


def ordinary_record(task_id, *, activity="report", workflow="wf1", campaign="c1", **fields):
    return {
        "task_id": task_id,
        "workflow_id": workflow,
        "campaign_id": campaign,
        "activity_id": activity,
        "subtype": "academy_action",
        "status": "FINISHED",
        **fields,
    }


def run(records):
    model = extract(from_dicts(records))
    return model, verify_all(model)


def findings_for(findings, check, status=None):
    return [
        finding for finding in findings
        if finding.check == check and (status is None or finding.status == status)
    ]


def test_only_completed_scientific_tool_calls_become_results():
    model, _ = run([
        calc("done", "CCO", 10.4),
        calc("failed", "CCC", 10.5, status="FAILED"),
        ordinary_record("llm", activity="llm_call", generated={"output": 99.0}),
    ])
    assert len(model.calcs) == 2
    assert len(model.results) == 1
    assert model.results[0].quantity == "ionization_energy"
    assert model.results[0].unit is None


def test_same_record_molecule_association_is_limited_and_traceable():
    model, findings = run([calc("t1", "CCO", 10.4)])
    association = findings_for(findings, "molecule_result_association", SUPPORTED)
    assert len(association) == 1
    assert association[0].source_record_ids == [model.dataset.records[0].uid]
    assert "does not independently validate" in association[0].limitations[0]


def test_conflicting_molecule_identifiers_remain_inconclusive():
    record = calc("t1", "CCO", 10.4)
    record["generated"]["smiles"] = "CCC"
    _, findings = run([record])
    assert findings_for(findings, "molecule_result_association", INCONCLUSIVE)


def test_equal_values_under_different_identifier_schemes_remain_ambiguous():
    record = calc("t1", "CCO", 10.4)
    record["used"]["molecule"] = "CCO"
    model, findings = run([record])
    assert model.calcs[0].identity_conflict
    assert findings_for(findings, "molecule_result_association", INCONCLUSIVE)


def test_numeric_consistency_requires_explicit_unit_and_matching_context():
    _, findings = run([
        calc("a", "CCO", 10.4, unit="eV"),
        calc("b", "CCO", 10.4 + 1e-11, unit="eV"),
    ])
    assert findings_for(findings, "numeric_consistency", SUPPORTED)

    _, findings = run([
        calc("a", "CCO", 10.4, with_protocol=False),
        calc("b", "CCO", 11.0, with_protocol=False),
    ])
    assert findings_for(findings, "numeric_consistency", INCONCLUSIVE)

    _, findings = run([
        calc("a", "CCO", 10.4, unit="eV", basis="method-a"),
        calc("b", "CCO", 9.0, unit="eV", basis="method-b"),
    ])
    assert not findings_for(findings, "numeric_consistency")


def test_repeated_numeric_outputs_require_an_exact_molecular_identifier():
    first = calc("a", "CCO", 10.4, unit="eV")
    second = calc("b", "CCO", 10.4, unit="eV")
    first["used"].pop("smiles")
    second["used"].pop("smiles")
    _, findings = run([first, second])
    hit = findings_for(findings, "numeric_consistency", INCONCLUSIVE)
    assert hit
    assert hit[0].reason_code == "IDENTITY_UNRESOLVED"


def test_unit_symbols_are_case_sensitive_and_unknown_units_are_not_compared():
    _, findings = run([
        calc("a", "CCO", 10.4, unit="eV"),
        calc("b", "CCO", 10.4, unit="EV"),
    ])
    hit = findings_for(findings, "numeric_consistency", INCONCLUSIVE)
    assert hit
    assert hit[0].reason_code == "UNIT_MISMATCH"


def test_nonmeasurement_numeric_fields_are_not_labeled_as_ionization_energy():
    record = calc("a", "CCO", 10.4, unit="eV")
    record["generated"]["confidence"] = 0.99
    model, _ = run([record])
    assert {result.quantity for result in model.results} == {
        "ionization_energy", "confidence",
    }


def test_claim_requires_known_molecule_quantity_and_matching_unit():
    claim = ordinary_record(
        "claim",
        generated={"conclusion": "CCO ionization energy is 8.1 eV."},
    )
    _, findings = run([calc("t1", "CCO", 10.4, unit="eV"), claim])
    assert findings_for(findings, "claim_evidence", CONTRADICTED)

    claim["generated"]["conclusion"] = "CCO ionization energy is 10.4 eV."
    _, findings = run([calc("t1", "CCO", 10.4, unit="eV"), claim])
    supported = findings_for(findings, "claim_evidence", SUPPORTED)
    assert supported
    result_evidence = supported[0].evidence[-1]["detail"]
    assert result_evidence["molecule_identifier_scheme"] == "smiles"
    assert result_evidence["molecule_identifier_field"] == "used.smiles"

    claim["generated"]["conclusion"] = "CCO ionization energy is 10.4."
    _, findings = run([calc("t1", "CCO", 10.4, unit="eV"), claim])
    assert findings_for(findings, "claim_evidence", INCONCLUSIVE)


def test_nonfinite_numeric_claim_is_not_reported_as_contradicted():
    claim = ordinary_record(
        "claim",
        generated={"conclusion": "CCO ionization energy is 1e999 eV."},
    )
    _, findings = run([calc("t1", "CCO", 10.4, unit="eV"), claim])
    hit = findings_for(findings, "claim_evidence", INCONCLUSIVE)
    assert hit
    assert hit[0].reason_code == "NONFINITE_CLAIM_VALUE"


def test_claims_do_not_match_results_from_another_workflow():
    claim = ordinary_record(
        "claim",
        workflow="claim-wf",
        generated={"conclusion": "CCO ionization energy is 10.4 eV."},
    )
    model, findings = run([
        calc("t1", "CCO", 10.4, unit="eV", workflow="calc-wf"),
        claim,
    ])
    hit = findings_for(findings, "claim_evidence", INCONCLUSIVE)
    assert hit
    assert hit[0].reason_code == "QUANTITY_UNRESOLVED"
    assert model.dataset.records[0].uid not in hit[0].source_record_ids


def test_claim_cannot_match_smiles_digits_as_the_scientific_value():
    claim = ordinary_record(
        "claim",
        generated={"conclusion": "C1CCCCC1 ionization energy is 7.2 eV."},
    )
    _, findings = run([calc("t1", "C1CCCCC1", 7.2, unit="eV"), claim])
    assert findings_for(findings, "claim_evidence", SUPPORTED)


def test_ie_abbreviation_is_extracted_as_one_atomic_claim():
    claim = ordinary_record(
        "claim",
        generated={
            "response": (
                "Goal: maximize ionization energy.\n"
                "CCO IE is 10.4 eV. Another unrelated sentence."
            ),
        },
    )
    model, findings = run([calc("t1", "CCO", 10.4, unit="eV"), claim])
    assert [item.text for item in model.claims] == ["CCO IE is 10.4 eV."]
    assert [item.text for item in model.objectives] == ["maximize ionization energy."]
    assert findings_for(findings, "claim_evidence", SUPPORTED)


def test_duplicate_claim_text_is_coalesced_without_losing_source_records():
    first = ordinary_record(
        "claim-a",
        generated={"conclusion": "CCO IE is 10.4 eV."},
    )
    second = ordinary_record(
        "claim-b",
        generated={"conclusion": "CCO IE is 10.4 eV."},
    )
    model, findings = run([calc("calc", "CCO", 10.4, unit="eV"), first, second])
    assert len(model.claims) == 1
    assert len(model.claims[0].source_refs) == 2
    claim_findings = findings_for(findings, "claim_evidence", SUPPORTED)
    assert len(claim_findings) == 1
    assert {
        model.claims[0].source_refs[0][0],
        model.claims[0].source_refs[1][0],
    }.issubset(claim_findings[0].source_record_ids)


def test_pairwise_comparison_is_not_misread_as_global_extremum():
    claim = ordinary_record(
        "claim",
        generated={"conclusion": "CCO has higher IE than CCC."},
    )
    _, findings = run([
        calc("a", "CCO", 11.0, unit="eV"),
        calc("b", "CCC", 10.0, unit="eV"),
        claim,
    ])
    assert findings_for(findings, "claim_evidence", SUPPORTED)

    _, findings = run([
        calc("a", "CCO", 9.0, unit="eV"),
        calc("b", "CCC", 10.0, unit="eV"),
        claim,
    ])
    assert findings_for(findings, "claim_evidence", CONTRADICTED)


def test_verb_comparison_is_parsed_as_a_pairwise_quantity_claim():
    claim = ordinary_record(
        "claim",
        generated={"conclusion": "CCO lowers IE relative to CCC."},
    )
    _, findings = run([
        calc("a", "CCO", 9.0, unit="eV"),
        calc("b", "CCC", 10.0, unit="eV"),
        claim,
    ])
    assert findings_for(findings, "claim_evidence", SUPPORTED)


def test_compound_comparison_is_not_forced_into_one_relation():
    claim = ordinary_record(
        "claim",
        generated={"conclusion": "CCO has higher IE than CCC but lower than CCN."},
    )
    _, findings = run([
        calc("a", "CCO", 11.0, unit="eV"),
        calc("b", "CCC", 10.0, unit="eV"),
        calc("c", "CCN", 12.0, unit="eV"),
        claim,
    ])
    hit = findings_for(findings, "claim_evidence", INCONCLUSIVE)
    assert hit
    assert hit[0].reason_code == "MULTIPLE_COMPARISONS"


def test_claim_with_chemical_names_is_not_mapped_to_smiles():
    claim = ordinary_record(
        "claim",
        generated={"conclusion": "Hydantoin has higher IE than methylurea."},
    )
    _, findings = run([
        calc("a", "CCO", 11.0, unit="eV"),
        calc("b", "CCC", 10.0, unit="eV"),
        claim,
    ])
    hit = findings_for(findings, "claim_evidence", INCONCLUSIVE)
    assert hit
    assert hit[0].reason_code == "IDENTITY_UNRESOLVED"


def test_numeric_claim_parser_does_not_take_unrelated_number_after_quantity():
    claim = ordinary_record(
        "claim",
        generated={"conclusion": "C1CCCCC1 ionization energy tends to exceed 7.2 in examples."},
    )
    model, findings = run([calc("a", "C1CCCCC1", 10.4, unit="eV"), claim])
    assert not model.claims
    assert model.claim_text_outside_supported_forms == 1
    assert not findings_for(findings, "claim_evidence")


def test_ranking_checks_independent_results_with_explicit_objective_and_units():
    objective = ordinary_record(
        "objective",
        activity="planner",
        workflow="report-wf",
        used={"objective": "Minimize ionization energy."},
    )
    ranking = ordinary_record(
        "rank",
        activity="report",
        workflow="report-wf",
        generated=[
            {"smiles": "CCO", "ionization_energy": 9.0},
            {"smiles": "CCC", "ionization_energy": 10.0},
        ],
    )
    ranking["custom_metadata"] = {"unit": "eV"}
    model, findings = run([
        calc("a", "CCO", 9.0, unit="eV", workflow="report-wf"),
        calc("b", "CCC", 10.0, unit="eV", workflow="report-wf"),
        objective,
        ranking,
        {**ranking, "task_id": "rank-duplicate"},
    ])
    assert len(model.rankings) == 1
    assert model.rankings[0].source_refs == [("task:line-4", 4), ("task:line-5", 5)]
    assert model.rankings[0].items[0]["molecule"] == "CCO"
    supported = findings_for(findings, "ranking_consistency", SUPPORTED)
    assert supported and {
        model.dataset.records[3].uid, model.dataset.records[4].uid,
    }.issubset(supported[0].source_record_ids)
    assert any("listed subset" in text for text in supported[0].limitations)

    model, findings = run([
        calc("a", "CCO", 9.0, unit="eV", workflow="report-wf"),
        calc("b", "CCC", 10.0, unit="eV", workflow="report-wf"),
        objective,
        {
            **ranking,
            "generated": [
                {"smiles": "CCC", "ionization_energy": 10.0},
                {"smiles": "CCO", "ionization_energy": 9.0},
            ],
        },
    ])
    assert findings_for(findings, "ranking_consistency", CONTRADICTED)


def test_ranking_without_its_own_unit_is_inconclusive():
    objective = ordinary_record(
        "objective",
        activity="planner",
        workflow="wf1",
        used={"objective": "Minimize ionization energy."},
    )
    ranking = ordinary_record(
        "rank",
        activity="report",
        workflow="wf1",
        generated=[
            {"smiles": "CCO", "ionization_energy": 9.0},
            {"smiles": "CCC", "ionization_energy": 10.0},
        ],
    )
    _, findings = run([
        calc("a", "CCO", 9.0, unit="eV"),
        calc("b", "CCC", 10.0, unit="eV"),
        objective,
        ranking,
    ])
    hit = findings_for(findings, "ranking_consistency", INCONCLUSIVE)
    assert hit
    assert hit[0].reason_code == "MISSING_RANKING_UNIT"


def test_ranking_stated_values_are_checked_against_matched_results():
    objective = ordinary_record(
        "objective",
        activity="planner",
        workflow="wf1",
        used={"objective": "Minimize ionization energy."},
    )
    ranking = ordinary_record(
        "rank",
        activity="report",
        workflow="wf1",
        generated=[
            {"smiles": "CCO", "ionization_energy": 9.0},
            {"smiles": "CCC", "ionization_energy": 10.0},
        ],
        custom_metadata={"unit": "eV"},
    )
    _, findings = run([
        calc("a", "CCO", 9.0, unit="eV"),
        calc("b", "CCC", 11.0, unit="eV"),
        objective,
        ranking,
    ])
    hit = findings_for(findings, "ranking_consistency", CONTRADICTED)
    assert hit
    assert hit[0].reason_code == "RANKING_VALUE_MISMATCH"


def test_ranking_without_independent_unit_or_objective_is_inconclusive():
    objective = ordinary_record(
        "objective",
        activity="planner",
        used={"objective": "Minimize ionization energy."},
    )
    ranking = ordinary_record(
        "rank",
        activity="report",
        generated=[["CCO", 9.0], ["CCC", 10.0]],
    )
    _, findings = run([
        calc("a", "CCO", 9.0),
        calc("b", "CCC", 10.0),
        objective,
        ranking,
    ])
    assert findings_for(findings, "ranking_consistency", INCONCLUSIVE)


def test_untyped_ranking_values_link_only_to_exact_same_workflow_identifiers():
    objective = ordinary_record(
        "objective",
        activity="planner",
        workflow="report-wf",
        used={"objective": "Minimize ionization energy."},
    )
    ranking = ordinary_record(
        "rank",
        activity="report",
        workflow="report-wf",
        generated=[["CCO", 9.0], ["CCC", 10.0]],
    )
    _, findings = run([
        calc("a", "CCO", 9.0, unit="eV", workflow="report-wf"),
        calc("b", "CCC", 10.0, unit="eV", workflow="report-wf"),
        objective,
        ranking,
    ])
    payload_match = findings_for(findings, "ranking_payload_consistency", SUPPORTED)
    assert payload_match
    assert payload_match[0].evidence_scope == "recorded_payload_consistency_only"
    assert any(
        item.get("detail", {}).get("exact_value_match_to_typed_identifier") == "CCO"
        for item in payload_match[0].evidence
        if isinstance(item.get("detail"), dict)
    )
    assert findings_for(findings, "ranking_consistency", INCONCLUSIVE)


def test_ranking_payload_mismatch_is_not_promoted_to_scientific_verdict():
    objective = ordinary_record(
        "objective",
        activity="planner",
        workflow="report-wf",
        used={"objective": "Minimize ionization energy."},
    )
    ranking = ordinary_record(
        "rank",
        activity="report",
        workflow="report-wf",
        generated=[["CCO", 9.0], ["CCC", 8.0]],
    )
    _, findings = run([
        calc("a", "CCO", 9.0, workflow="report-wf"),
        calc("b", "CCC", 10.0, workflow="report-wf"),
        objective,
        ranking,
    ])
    payload_mismatch = findings_for(
        findings, "ranking_payload_consistency", CONTRADICTED
    )
    assert payload_mismatch
    assert payload_mismatch[0].reason_code == "RANKING_PAYLOAD_VALUE_MISMATCH"
    assert findings_for(findings, "ranking_consistency", INCONCLUSIVE)


def test_ranking_with_missing_molecule_identifier_is_inconclusive():
    objective = ordinary_record(
        "objective",
        activity="planner",
        workflow="report-wf",
        used={"objective": "Minimize ionization energy."},
    )
    ranking = ordinary_record(
        "rank",
        activity="report",
        workflow="report-wf",
        generated=[
            {"smiles": "CCO", "ionization_energy": 9.0},
            {"ionization_energy": 10.0},
        ],
        custom_metadata={"unit": "eV"},
    )
    _, findings = run([
        calc("a", "CCO", 9.0, unit="eV", workflow="report-wf"),
        calc("b", "CCC", 10.0, unit="eV", workflow="report-wf"),
        objective,
        ranking,
    ])
    hit = findings_for(findings, "ranking_consistency", INCONCLUSIVE)
    assert hit
    assert hit[0].reason_code == "MISSING_RANKING_IDENTIFIER"
    assert not findings_for(findings, "ranking_payload_consistency")


def test_untyped_unmatched_ranking_labels_are_not_guessed():
    objective = ordinary_record(
        "objective",
        activity="planner",
        workflow="report-wf",
        used={"objective": "Minimize ionization energy."},
    )
    ranking = ordinary_record(
        "rank",
        activity="report",
        workflow="report-wf",
        generated=[["ethanol", 9.0], ["chloroethane", 10.0]],
        custom_metadata={"unit": "eV"},
    )
    _, findings = run([
        calc("a", "CCO", 9.0, unit="eV", workflow="report-wf"),
        calc("b", "CCCl", 10.0, unit="eV", workflow="report-wf"),
        objective,
        ranking,
    ])
    hit = findings_for(findings, "ranking_consistency", INCONCLUSIVE)
    assert hit
    assert hit[0].reason_code == "IDENTIFIER_SCHEME_MISSING"
    assert not findings_for(findings, "ranking_payload_consistency")


def test_duplicate_ranking_identifiers_are_not_compared():
    objective = ordinary_record(
        "objective",
        activity="planner",
        workflow="report-wf",
        used={"objective": "Minimize ionization energy."},
    )
    ranking = ordinary_record(
        "rank",
        activity="report",
        workflow="report-wf",
        generated=[["CCO", 9.0], ["CCO", 9.0]],
        custom_metadata={"unit": "eV"},
    )
    _, findings = run([
        calc("a", "CCO", 9.0, unit="eV", workflow="report-wf"),
        objective,
        ranking,
    ])
    hit = findings_for(findings, "ranking_consistency", INCONCLUSIVE)
    assert hit
    assert hit[0].reason_code == "DUPLICATE_RANKING_IDENTIFIER"
    assert not findings_for(findings, "ranking_payload_consistency")


def test_ranking_does_not_match_calculations_from_another_workflow():
    ranking = ordinary_record(
        "rank",
        activity="report",
        workflow="rank-wf",
        generated=[["CCO", 9.0], ["CCC", 10.0]],
    )
    _, findings = run([
        calc("a", "CCO", 9.0, workflow="calc-wf"),
        calc("b", "CCC", 10.0, workflow="calc-wf"),
        ranking,
    ])
    hit = findings_for(findings, "ranking_consistency", INCONCLUSIVE)
    assert hit
    assert hit[0].reason_code == "NO_SCOPED_CALCULATION_RESULTS"
    assert not findings_for(findings, "ranking_payload_consistency")


def test_ranking_payload_check_is_executed_and_reported_separately(tmp_path):
    objective = ordinary_record(
        "objective",
        activity="planner",
        workflow="report-wf",
        used={"objective": "Minimize ionization energy."},
    )
    ranking = ordinary_record(
        "rank",
        activity="report",
        workflow="report-wf",
        generated=[["CCO", 9.0], ["CCC", 10.0]],
    )
    model, findings = run([
        calc("a", "CCO", 9.0, unit="eV", workflow="report-wf"),
        calc("b", "CCC", 10.0, unit="eV", workflow="report-wf"),
        objective,
        ranking,
    ])
    assert findings_for(findings, "ranking_payload_consistency", SUPPORTED)
    assert findings_for(findings, "ranking_consistency", INCONCLUSIVE)

    paths = write_reports(model, findings, tmp_path, "input.jsonl", 1e-9)
    report = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert report["recorded_payload_consistency"]["SUPPORTED"] == 1
    assert report["trace_consistency_checks"]["SUPPORTED"] == 0
    assert any(
        finding["check"] == "ranking_payload_consistency"
        and finding["evidence_scope"] == "recorded_payload_consistency_only"
        for finding in report["findings"]
    )
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "| Raw ranking-value copies | 1 |" in markdown


def test_gaied_shaped_record_does_not_infer_unit_or_scientific_protocol():
    record = calc("gaied", "NC(N)=O", 14.959175581136662, with_protocol=False)
    record["used"] = {"input": "{'smiles': 'NC(N)=O'}"}
    record["generated"] = {"output": 14.959175581136662}
    model, findings = run([record])
    assert len(model.results) == 1
    result = model.results[0]
    assert result.molecule == "NC(N)=O"
    assert result.unit is None
    assert result.scientific_protocol is None
    protocol = findings_for(findings, "calculation_protocol", INCONCLUSIVE)
    assert len(protocol) == 1
    assert protocol[0].reason_code == "MISSING_CALCULATION_PROTOCOL"


def test_complete_protocol_is_reported_as_completeness_not_validation():
    _, findings = run([calc("a", "CCO", 9.0, unit="Ha")])
    finding = findings_for(findings, "calculation_protocol", SUPPORTED)[0]
    assert "complete, internally coherent" in finding.message
    assert finding.evidence_scope == "protocol_completeness_only"
    assert any("does not independently validate" in limitation for limitation in finding.limitations)


def test_protocol_unit_conflict_is_explicitly_contradicted():
    record = calc("a", "CCO", 9.0, unit="eV")
    record["custom_metadata"]["scientific_protocol"]["energy_unit"] = "Ha"
    _, findings = run([record])
    finding = findings_for(findings, "calculation_protocol", CONTRADICTED)[0]
    assert finding.reason_code == "PROTOCOL_UNIT_MISMATCH"


def test_protocol_mismatch_blocks_numeric_comparison_and_ionization_charge_conflict():
    first = calc("a", "CCO", 10.4, unit="eV")
    second = calc("b", "CCO", 10.5, unit="eV")
    second["custom_metadata"]["scientific_protocol"]["method"] = "different-method"
    _, findings = run([first, second])
    hit = findings_for(findings, "numeric_consistency", INCONCLUSIVE)
    assert hit
    assert hit[0].reason_code == "INCOMPARABLE_CALCULATION_PROTOCOL"

    invalid = calc("bad-charge", "CCO", 10.4, unit="eV")
    invalid["custom_metadata"]["scientific_protocol"]["cation_charge"] = 0
    _, findings = run([invalid])
    hit = findings_for(findings, "calculation_protocol", CONTRADICTED)
    assert hit
    assert hit[0].reason_code == "IONIZATION_CHARGE_MISMATCH"

    conflict = calc("conflict", "CCO", 10.4, unit="eV")
    conflict["used"]["scientific_protocol"] = {
        **conflict["custom_metadata"]["scientific_protocol"],
        "software_version": "other-version",
    }
    _, findings = run([conflict])
    hit = findings_for(findings, "calculation_protocol", INCONCLUSIVE)
    assert hit
    assert hit[0].reason_code == "CONFLICTING_CALCULATION_PROTOCOL"


def test_missing_protocol_field_and_nonconverged_geometry_are_not_supported():
    incomplete = calc("incomplete", "CCO", 10.4, unit="eV")
    del incomplete["custom_metadata"]["scientific_protocol"]["software_version"]
    _, findings = run([incomplete])
    hit = findings_for(findings, "calculation_protocol", INCONCLUSIVE)
    assert hit
    assert hit[0].reason_code == "INCOMPLETE_CALCULATION_PROTOCOL"

    nonconverged = calc("nonconverged", "CCO", 10.4, unit="eV")
    nonconverged["custom_metadata"]["scientific_protocol"][
        "all_geometry_optimizations_converged"
    ] = False
    _, findings = run([nonconverged])
    hit = findings_for(findings, "calculation_protocol", INCONCLUSIVE)
    assert hit
    assert hit[0].reason_code == "CALCULATION_NOT_CONVERGED"


def test_numeric_claim_is_not_scientifically_supported_without_protocol():
    claim = ordinary_record(
        "claim",
        generated={"conclusion": "CCO IE is 10.4 eV."},
    )
    _, findings = run([
        calc("t1", "CCO", 10.4, unit="eV", with_protocol=False),
        claim,
    ])
    hit = findings_for(findings, "claim_evidence", INCONCLUSIVE)
    assert hit
    assert hit[0].reason_code == "MISSING_CALCULATION_PROTOCOL"
    assert hit[0].evidence_scope == "recorded_trace_consistency"


def test_workflow_extremum_is_inconclusive_without_complete_candidate_set():
    claim = ordinary_record(
        "claim",
        generated={
            "conclusion": (
                "CCO has the highest ionization energy among tested molecules."
            ),
        },
    )
    _, findings = run([
        calc("a", "CCO", 11.0, unit="eV"),
        calc("b", "CCC", 10.0, unit="eV"),
        claim,
    ])
    hit = findings_for(findings, "claim_evidence", INCONCLUSIVE)
    assert hit
    assert hit[0].reason_code == "CANDIDATE_SET_COMPLETENESS_UNVERIFIED"


def test_explicit_parent_links_are_resolved_in_graph_and_report():
    model, findings = run([
        ordinary_record("parent", activity="setup"),
        calc("child", "CCO", 10.4, parent_task_id="parent"),
    ])
    child = next(record for record in model.dataset.records if record.original_id == "child")
    parent = next(record for record in model.dataset.records if record.original_id == "parent")
    assert child.parent_references == [("parent_task_id", "parent")]
    edge_data = model.graph.get_edge_data(f"evidence::{child.uid}", f"evidence::{parent.uid}")
    assert edge_data and any(edge["relation"] == "derived_from" for edge in edge_data.values())
    parent_finding = findings_for(findings, "parent_reference", SUPPORTED)[0]
    assert any(
        evidence["field"] == "parent_task_id"
        for evidence in parent_finding.evidence
    )
    assert any(
        edge["source_fields"] == ["parent_task_id"]
        for edge in edge_data.values()
        if edge["relation"] == "derived_from"
    )


def test_parent_reference_source_path_preserves_nested_and_overridden_fields():
    dataset = from_dicts([
        {
            "task_id": "nested-only",
            "data": {"parent_task_id": "nested-parent"},
        },
        {
            "task_id": "overridden",
            "data": {"parent_task_id": "shadowed-parent"},
            "parent_task_id": "top-level-parent",
        },
    ])
    nested, overridden = dataset.records
    assert nested.parent_references == [
        ("data.parent_task_id", "nested-parent"),
    ]
    assert overridden.parents == ["top-level-parent"]
    assert overridden.parent_references == [
        ("parent_task_id", "top-level-parent"),
    ]


def test_provenance_cycle_evidence_uses_actual_parent_fields():
    _, findings = run([
        ordinary_record("first", parent_task_id="second"),
        ordinary_record("second", parent_task_id="first"),
    ])
    cycle = findings_for(findings, "provenance_cycle", CONTRADICTED)
    assert cycle
    assert {
        evidence["field"] for evidence in cycle[0].evidence
    } == {"parent_task_id"}
    assert {
        evidence["detail"] for evidence in cycle[0].evidence
    } == {"first", "second"}


def test_duplicate_ids_make_parent_resolution_inconclusive():
    _, findings = run([
        ordinary_record("parent"),
        ordinary_record("parent"),
        calc("child", "CCO", 10.4, parent_task_id="parent"),
    ])
    parent_findings = findings_for(findings, "parent_reference")
    assert parent_findings
    assert all(finding.status == INCONCLUSIVE for finding in parent_findings)
    assert all(
        finding.reason_code == "PARENT_REFERENCE_AMBIGUOUS"
        for finding in parent_findings
    )
    assert all(
        evidence["field"] == "parent_task_id"
        for finding in parent_findings
        for evidence in finding.evidence
    )


def test_missing_parent_record_is_distinguished_from_ambiguous_parent():
    model, findings = run([
        calc("child", "CCO", 10.4, parent_task_id="not-exported"),
    ])
    hit = findings_for(findings, "parent_reference", INCONCLUSIVE)
    assert hit
    assert hit[0].reason_code == "PARENT_ID_NOT_FOUND_IN_EXPORT"
    assert "not found" in hit[0].message
    assert any(
        evidence["field"] == "parent_task_id"
        and evidence["detail"] == "not-exported"
        for evidence in hit[0].evidence
    )
    assert model.dataset.records[0].parents == ["not-exported"]


def test_malformed_jsonl_is_reported_not_silently_discarded():
    dataset = parse_lines([
        '{"task_id":"x","workflow_id":"w","campaign_id":"c"}',
        "not json",
        "[1,2]",
    ])
    assert len(dataset.records) == 1
    assert len(dataset.parse_errors) == 2
    findings = verify_all(extract(dataset))
    assert len(findings_for(findings, "input_parse_error", INCONCLUSIVE)) == 2


def test_report_schema_and_graph_have_traceable_sources(tmp_path):
    model, findings = run([calc("t1", "CCO", 10.4)])
    paths = write_reports(model, findings, tmp_path, "input.jsonl", 1e-9)
    report = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert report["coverage"]["finite_results"] == 1
    assert report["coverage"]["results_without_units"] == 1
    assert report["coverage"]["results_with_complete_protocol"] == 0
    assert report["coverage"]["rankings_without_same_workflow_results"] == 0
    assert report["independent_scientific_validation"]["performed"] is False
    assert report["recorded_payload_consistency"]["SUPPORTED"] == 0
    assert report["assessment_summary"]["performed"] > 0
    assert report["findings"][0]["id"]
    assert report["findings"][0]["assessment"] == "performed"
    assert report["findings"][0]["reason_code"]
    assert report["findings"][0]["source_record_ids"]
    assert report["findings"][0]["evidence_scope"] == "provenance_or_protocol_completeness"
    assert report["findings"][0]["evidence"][0]["source_ref"]["source_type"] == "local_jsonl"
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert len(markdown) < 10_000
    assert "no claim was independently validated" in markdown
    assert "verification_report.json" in markdown
    assert "### 1." not in markdown
    graph = json.loads(Path(paths["graph"]).read_text(encoding="utf-8"))
    assert any(node["type"] == "ScientificResult" for node in graph["nodes"])
    assert any(edge["relation"] == "supported_by" for edge in graph["edges"])


def test_report_lists_unresolved_parent_reference_source_fields(tmp_path):
    model, findings = run([
        calc("child", "CCO", 10.4, parent_task_id="outside-export"),
    ])
    paths = write_reports(model, findings, tmp_path, "input.jsonl", 1e-9)
    report = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    audit = report["parent_reference_audit"]
    assert audit["unresolved_count"] == 1
    assert audit["unresolved_distinct_id_count"] == 1
    assert audit["ambiguous_count"] == 0
    assert audit["unresolved"][0]["source_field"] == "parent_task_id"
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "Not found does not mean invalid" in markdown
    assert "`parent_task_id`" in markdown


def test_graph_keeps_different_identifier_schemes_as_distinct_molecules():
    smiles_record = calc("smiles", "CCO", 10.4)
    label_record = calc("label", "CCO", 10.4)
    label_record["used"] = {"molecule": "CCO", "basis": "xtb"}
    model, _ = run([smiles_record, label_record])
    molecule_nodes = [
        node for node, attrs in model.graph.nodes(data=True)
        if attrs.get("type") == "Molecule"
    ]
    assert "molecule::c1::smiles::CCO" in molecule_nodes
    assert "molecule::c1::molecule::CCO" in molecule_nodes
