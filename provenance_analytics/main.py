"""Command-line entry point for Provenance Analytics.

Usage:
    python main.py --input path/to/provenance.jsonl
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from knowledge_graph import extract
from provenance import load_jsonl
from verification import (
    verify_all,
    write_reports,
    summarize,
    summarize_assessments,
    summarize_scientific_checks,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify scientific provenance from a JSONL trace."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the input provenance JSONL file.",
    )
    parser.add_argument(
        "--output-dir", default=str(Path(__file__).resolve().parent / "output"),
        help="Directory for generated reports (default: ./output).",
    )
    parser.add_argument(
        "--rel-tol", type=float, default=1e-9,
        help="Relative tolerance for repeated numerical values (default: 1e-9).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not math.isfinite(args.rel_tol) or args.rel_tol < 0:
        parser.error("--rel-tol must be a finite non-negative number")

    input_path = Path(args.input).expanduser()
    if not input_path.is_file():
        print(f"ERROR: input file does not exist: {input_path}", file=sys.stderr)
        return 2

    try:
        dataset = load_jsonl(input_path)
        model = extract(dataset)
        findings = verify_all(model, args.rel_tol)
        paths = write_reports(
            model=model,
            findings=findings,
            output_dir=args.output_dir,
            input_path=str(input_path.resolve()),
            rel_tol=args.rel_tol,
        )
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    counts = summarize(findings)

    print("Provenance Analytics completed.")
    print(f"Input records: {len(dataset.records)}")
    print(f"Malformed lines: {len(dataset.parse_errors)}")
    print(f"Findings: {len(findings)}")
    print(
        "Status counts: "
        f"SUPPORTED={counts['SUPPORTED']}, "
        f"CONTRADICTED={counts['CONTRADICTED']}, "
        f"INCONCLUSIVE={counts['INCONCLUSIVE']}"
    )
    scientific = summarize_scientific_checks(findings)
    print(
        "Claim/ranking/repeatability trace-consistency checks: "
        f"SUPPORTED={scientific['SUPPORTED']}, "
        f"CONTRADICTED={scientific['CONTRADICTED']}, "
        f"INCONCLUSIVE={scientific['INCONCLUSIVE']}"
    )
    payload_checks = summarize([
        finding for finding in findings
        if finding.check == "ranking_payload_consistency"
    ])
    print(
        "Raw ranking payload checks (not scientific validation): "
        f"SUPPORTED={payload_checks['SUPPORTED']}, "
        f"CONTRADICTED={payload_checks['CONTRADICTED']}, "
        f"INCONCLUSIVE={payload_checks['INCONCLUSIVE']}"
    )
    assessments = summarize_assessments(findings)
    print(
        "Assessment completeness: "
        f"performed={assessments['performed']}, "
        f"not_performed={assessments['not_performed']}, "
        f"indeterminate={assessments['indeterminate']}"
    )
    print("Reports:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
