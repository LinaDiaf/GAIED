"""
Read or edit the campaign record of a memory: the goal and settings that every agent and the analysis agent receive
at the top of their memory section.

    python -m pgmem.campaign                              print the campaign
    python -m pgmem.campaign set goal "find molecules with the highest ionization energy"
    python -m pgmem.campaign unset FIELD

Namespace: --namespace or PGMEM_NAMESPACE (default "default"). A goal set before the campaign starts is kept: the
agents only write the fields that are absent.
"""
from __future__ import annotations

import argparse
import os

from .store import MemoryStore


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", choices=["show", "set", "unset"], default="show")
    ap.add_argument("field", nargs="?")
    ap.add_argument("value", nargs="?")
    ap.add_argument("--namespace", default=os.environ.get("PGMEM_NAMESPACE", "default"))
    a = ap.parse_args(argv)
    s = MemoryStore.connect(os.environ.get("REDIS_HOST", "localhost"), int(os.environ.get("REDIS_PORT", "6379")),
                            a.namespace)
    if a.command == "set":
        if not a.field or a.value is None:
            ap.error("set needs FIELD and VALUE")
        s.update_campaign(a.field, a.value)
    elif a.command == "unset":
        if not a.field:
            ap.error("unset needs FIELD")
        s.update_campaign(a.field, None)
    print(s.campaign_text() or f"namespace {a.namespace}: no campaign recorded")


if __name__ == "__main__":
    main()
