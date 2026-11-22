"""
Which motifs a model proposes, to choose the failing motifs of the motif_failures scenario (an injected motif that
the model rarely proposes leaves nothing to learn from).

    python -m evaluation.motif_frequency results/eval/pilot      -> share of the valid proposals with each motif

Reads the proposals (tool calls of compute_ionization_energy) in the Flowcept provenance of every run of a series.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter

from pgmem import chem
from pgmem.provenance import smiles_of


def main(series_dir: str) -> None:
    motifs, valid, invalid = Counter(), 0, 0
    for path in sorted(glob.glob(os.path.join(series_dir, "*", "provenance.jsonl"))):
        seen = set()
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("subtype") != "tool_call" or r.get("activity_id") != "compute_ionization_energy" \
                    or r.get("task_id") in seen:
                continue
            seen.add(r.get("task_id"))
            smiles = smiles_of(r) or ""
            if not chem.parses(smiles):
                invalid += 1
                continue
            valid += 1
            motifs.update(chem.features(smiles) - chem.GENERIC)
    total = valid + invalid
    print(f"{total} proposals: {valid} valid, {invalid} SMILES RDKit cannot read ({invalid / max(total, 1):.0%})")
    print("share of the valid proposals with each motif (generic groups left out):")
    for m, n in motifs.most_common():
        print(f"  {m:28s} {n / max(valid, 1):5.0%}  ({n})  SMARTS {chem.FEATURES[m]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/eval/pilot")
