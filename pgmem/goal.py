"""
Read or set the campaign goal kept in a memory, which the analysis agent uses to check that the agents still pursue
it (goal drift) and to start the prompts of its explanations, procedures and advice.

    python -m pgmem.goal                                   print the goal
    python -m pgmem.goal set goal "find molecules with high ionization energy"
    python -m pgmem.goal set direction maximize            maximize | minimize

Namespace: --namespace or PGMEM_NAMESPACE (default "default"). A goal set before the campaign starts is kept: the
agents and the analysis agent only write it (PGMEM_GOAL, PGMEM_DIRECTION) when the memory has none. The agents
themselves receive their goal from GAIED's own prompt: keep both the same.
"""
from __future__ import annotations

import argparse
import os

from .store import GOAL_FIELDS, MemoryStore


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", choices=["show", "set"], default="show")
    ap.add_argument("field", nargs="?", choices=GOAL_FIELDS)
    ap.add_argument("value", nargs="?")
    ap.add_argument("--namespace", default=os.environ.get("PGMEM_NAMESPACE", "default"))
    a = ap.parse_args(argv)
    s = MemoryStore.connect(os.environ.get("REDIS_HOST", "localhost"), int(os.environ.get("REDIS_PORT", "6379")),
                            a.namespace)
    if a.command == "set":
        if not a.field or a.value is None:
            ap.error("set needs FIELD and VALUE")
        s.update_goal(a.field, a.value)
    print(s.goal_text() or f"namespace {a.namespace}: no goal recorded")


if __name__ == "__main__":
    main()
