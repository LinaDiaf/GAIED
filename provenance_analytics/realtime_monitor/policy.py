"""Conservative, evidence-bound remediation rules."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

INSTRUCTIONS = {
    "INPUT_PARSE_ERROR": (
        "The recorded compute_ionization_energy task failed because RDKit could not parse the submitted SMILES, so no xTB energy exists. Do not claim a result for it. Retry with a neutral, RDKit-valid SMILES (do not append a bare '+'); the tool performs the ionization-energy calculation. Record the corrected candidate and its actual result."
    ),
    "PARENT_ID_NOT_FOUND_IN_EXPORT": (
        "This parent reference is not present in the records received so far. "
        "Provide the actual referenced record if available, or explicitly mark "
        "the link as outside this stream; do not infer a parent."
    ),
    "PARENT_REFERENCE_AMBIGUOUS": (
        "This parent ID matches multiple received records. Provide an explicit "
        "unique source identifier or leave the relationship unresolved; do not guess."
    ),
    "PARENT_WORKFLOW_NOT_IN_CAMPAIGN": (
        "This parent workflow was not found in the records received for this "
        "campaign. Provide its actual workflow record or mark it as external; "
        "do not infer a link."
    ),
    "PARENT_WORKFLOW_AMBIGUOUS": (
        "This parent workflow ID matches multiple records in the campaign. "
        "Provide a unique explicit workflow reference or leave the link unresolved."
    ),
    "MISSING_UNIT": (
        "The comparison lacks an explicit unit. Emit the unit recorded by the "
        "calculation, or leave the comparison unresolved; do not guess or convert it."
    ),
    "UNIT_MISMATCH": (
        "The conclusion labels the recorded ionization-energy value with a different unit than the calculation record. Correct the conclusion to use the recorded unit exactly; do not convert the numeric value unless you perform and record an explicit conversion."
    ),
    "MISSING_CALCULATION_PROTOCOL": (
        "The calculation protocol is absent. Emit only protocol fields recorded "
        "by the actual execution; do not fill fields with defaults."
    ),
    "INCOMPLETE_CALCULATION_PROTOCOL": (
        "Some calculation protocol fields are missing. Provide them only if "
        "recorded by the actual execution; otherwise keep the comparison unresolved."
    ),
    "IDENTITY_UNRESOLVED": (
        "The molecule identity is unresolved. Provide the exact identifier and "
        "its scheme from the source record; do not map names to structures by assumption."
    ),
    "NO_MATCHING_RESULT": (
        "No matching result is present in this workflow scope. Provide the "
        "source result record if it exists, or leave the claim unverified; do not "
        "borrow a result from another workflow."
    ),
    "MISSING_RESULT_EVIDENCE": (
        "This calculation has no extractable finite output. Provide the actual "
        "output record or its failure state; do not invent a result."
    ),
    "RESULT_OUTPUT_MISSING": (
        "This calculation has no extractable finite output. Provide the actual "
        "output record or its failure state; do not invent a result."
    ),
    "CHECK_CONFLICT": (
        "The assertion conflicts with the cited trace values. Review the cited "
        "records and either correct the assertion or provide additional source "
        "evidence; agreement with a trace alone does not establish scientific truth."
    ),
    "RANKING_ORDER_MISMATCH": (
        "The recorded ranking order conflicts with the available same-workflow "
        "outputs and objective. Review the cited records; do not claim the ranking "
        "is scientifically validated."
    ),
    "RANKING_VALUE_MISMATCH": (
        "A ranking value differs from its uniquely matched same-workflow output. "
        "Check the source records and correct the copy or explain the discrepancy."
    ),
}


def create_intervention(
    run_id: str,
    finding: dict[str, Any],
    event_id: str,
    *,
    run_completed: bool = False,
) -> dict[str, Any] | None:
    status = finding.get("status")
    reason = finding.get("reason_code")
    check = finding.get("check")
    if status not in {"INCONCLUSIVE", "CONTRADICTED"}:
        return None
    if reason == "NO_MATCHING_RESULT" and not run_completed:
        return None
    if reason == "RESULT_OUTPUT_MISSING":
        terminal_statuses = {
            "finished", "succeeded", "success", "completed", "complete", "ok",
            "failed", "error", "cancelled", "canceled", "timeout",
        }
        source_statuses = [
            item.get("detail", {}).get("status")
            for item in finding.get("evidence", [])
            if isinstance(item.get("detail"), dict)
        ]
        if not source_statuses or any(
            not isinstance(source_status, str)
            or source_status.strip().casefold() not in terminal_statuses
            for source_status in source_statuses
        ):
            return None
    instruction = INSTRUCTIONS.get(reason) if isinstance(reason, str) else None
    if instruction is None and status == "CONTRADICTED" and check == "claim_evidence":
        instruction = INSTRUCTIONS["CHECK_CONFLICT"]
    if instruction is None:
        return None
    finding_id = finding.get("id")
    if not isinstance(finding_id, str) or not finding_id:
        raise ValueError("cannot intervene on a finding without a stable ID")
    identity = f"{run_id}\0{finding_id}\0{event_id}".encode("utf-8")
    return {
        "intervention_id": hashlib.sha256(identity).hexdigest()[:20],
        "run_id": run_id,
        "finding_id": finding_id,
        "check": check,
        "scope": finding.get("scope"),
        "reason_code": reason,
        "instruction": instruction,
        "evidence": finding.get("evidence", []),
        "source_record_ids": finding.get("source_record_ids", []),
        "status": "awaiting_agent_response",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trigger_event_id": event_id,
    }
