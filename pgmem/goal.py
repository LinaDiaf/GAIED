"""
The campaign goal kept in a memory, for the analysis agent, and what "better" means for a result.

The goal is a text ("find molecules with high ionization energy", PGMEM_GOAL): it starts the prompts of the analysis
agent's explanations, procedures and advice, and the plans of the agents are read against it (goal drift). The
agents themselves receive their goal from GAIED's own prompt, not from the memory: keep both the same.

"Better" is defined in one place, `score`: the campaign maximizes the measured quantity. Another objective changes
this function only.

    python -m pgmem.goal                       print the goal of a memory
    python -m pgmem.goal set "find molecules with high ionization energy"

Namespace: --namespace or PGMEM_NAMESPACE (default "default"). A goal set before the campaign starts is kept: the
agents and the analysis agent write PGMEM_GOAL only when the memory has none. The goal is fixed for a campaign.
"""
from __future__ import annotations

import argparse
import os
import re


def score(value: float) -> float:
    """Higher is better: the campaign maximizes the measured quantity."""
    return value


def ranked(results: list) -> list:
    """M1 results with a value, best first."""
    return sorted((e for e in results if e.data.get("value") is not None), key=lambda e: -score(e.data["value"]))


# ---------------------------------------------------------------------- quotes
_MARKUP = re.compile(r"[*_`#>\"'“”‘’]")
_NON_WORD = re.compile(r"[^\w.]+|(?<!\d)\.|\.(?!\d)")   # keeps decimal points, drops other punctuation
_ELLIPSIS = re.compile(r"\.\.\.|…")
MIN_QUOTE = 12        # characters of a normalized quote: shorter ones ("IE") would match almost any plan


def _normal(text: str) -> str:
    return " ".join(_NON_WORD.sub(" ", _MARKUP.sub(" ", text.lower())).split())


def quoted_in(quote: str, text: str) -> bool:
    """Whether `quote` was written in `text`: the same words in the same order once case, markup, punctuation and
    spacing are set aside; a quote cut with "..." matches when each of its parts appears, in order. No similarity
    threshold: one word changed (a "raise" quoted as "lower") is not a match."""
    parts = [p for p in (_normal(x) for x in _ELLIPSIS.split(quote or "")) if p]
    if sum(len(p) for p in parts) < MIN_QUOTE:
        return False
    hay, start = f" {_normal(text or '')} ", 0
    for p in parts:
        i = hay.find(f" {p} ", start)
        if i < 0:
            return False
        start = i + len(p) + 1
    return True


def main(argv: list[str] | None = None) -> None:
    from .store import MemoryStore
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", choices=["show", "set"], default="show")
    ap.add_argument("text", nargs="?")
    ap.add_argument("--namespace", default=os.environ.get("PGMEM_NAMESPACE", "default"))
    a = ap.parse_args(argv)
    s = MemoryStore.connect(os.environ.get("REDIS_HOST", "localhost"), int(os.environ.get("REDIS_PORT", "6379")),
                            a.namespace)
    if a.command == "set":
        if not a.text:
            ap.error("set needs the goal text")
        s.update_goal(a.text)
    print(s.goal_text() or f"namespace {a.namespace}: no goal recorded")


if __name__ == "__main__":
    main()
