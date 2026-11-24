"""
The campaign goal kept in a memory, for the analysis agent: what the agents must pursue, in words, and, when the goal
is a target quantity, in a form the code can measure results against.

  text        the goal in words (any goal: a property, constraints, an exploration aim)
  quantity    the measured quantity it is about (M1 `quantity`), if any
  direction   maximize | minimize | target
  target      the target value or interval ("10" or "9,11") when the direction is "target"

The analysis agent uses it to judge whether an agent still pursues the goal (goal drift: its results moving away
from it, its plan and conclusions read by the LLM against it) and to start the prompts of its explanations,
procedures and advice. The agents receive their goal from GAIED's own prompt, not from the memory: keep both the same.

    python -m pgmem.goal                                   print the goal of a memory
    python -m pgmem.goal set text "find molecules with high ionization energy"
    python -m pgmem.goal set direction target
    python -m pgmem.goal set target 9,11

Namespace: --namespace or PGMEM_NAMESPACE (default "default"). A goal set before the campaign starts is kept: the
agents and the analysis agent only write the fields the memory does not have (PGMEM_GOAL, PGMEM_GOAL_QUANTITY,
PGMEM_DIRECTION, PGMEM_TARGET).
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass

DIRECTIONS = ("maximize", "minimize", "target")
FIELDS = ("text", "quantity", "direction", "target")


def parse_target(text: str | None) -> tuple[float, float] | None:
    """"10" -> (10, 10); "9,11" -> (9, 11); None or "" -> None."""
    if not text:
        return None
    values = sorted(float(x) for x in str(text).split(","))
    if len(values) not in (1, 2):
        raise ValueError(f"a target is a value or an interval 'low,high', got {text!r}")
    return values[0], values[-1]


@dataclass
class Goal:
    text: str
    quantity: str | None = None
    direction: str | None = None
    target: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.direction is not None and self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {self.direction!r}")

    @property
    def measurable(self) -> bool:
        """Whether results can be scored against the goal."""
        return bool(self.quantity and self.direction and (self.direction != "target" or self.target))

    def score(self, value: float) -> float:
        """Higher is closer to the goal: the value, its opposite, or minus its distance to the target interval."""
        if self.direction == "maximize":
            return value
        if self.direction == "minimize":
            return -value
        low, high = self.target
        return -max(low - value, 0.0, value - high)

    def ranked(self, results: list) -> list:
        """M1 results of the goal's quantity, closest to the goal first ([] when the goal is not measurable)."""
        if not self.measurable:
            return []
        return sorted((e for e in results if e.data.get("quantity") == self.quantity and e.data.get("value") is not None),
                      key=lambda e: -self.score(e.data["value"]))

    def describe(self) -> str:
        how = ""
        if self.measurable:
            how = (f" (target: {self.quantity} in [{self.target[0]:g}, {self.target[1]:g}])"
                   if self.direction == "target" else f" ({self.direction} {self.quantity})")
        return f"Campaign goal: {self.text}{how}"

    def to_fields(self) -> dict:
        target = None if self.target is None else (
            f"{self.target[0]:g}" if self.target[0] == self.target[1] else f"{self.target[0]:g},{self.target[1]:g}")
        return {f: v for f, v in (("text", self.text), ("quantity", self.quantity), ("direction", self.direction),
                                  ("target", target)) if v}

    @classmethod
    def from_fields(cls, f: dict) -> "Goal | None":
        if not f.get("text"):
            return None
        return cls(f["text"], f.get("quantity") or None, f.get("direction") or None, parse_target(f.get("target")))


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
    ap.add_argument("field", nargs="?", choices=FIELDS)
    ap.add_argument("value", nargs="?")
    ap.add_argument("--namespace", default=os.environ.get("PGMEM_NAMESPACE", "default"))
    a = ap.parse_args(argv)
    s = MemoryStore.connect(os.environ.get("REDIS_HOST", "localhost"), int(os.environ.get("REDIS_PORT", "6379")),
                            a.namespace)
    if a.command == "set":
        if not a.field or a.value is None:
            ap.error("set needs FIELD and VALUE")
        s.update_goal(a.field, a.value)
    g = s.goal()
    print(g.describe() if g else f"namespace {a.namespace}: no goal recorded")


if __name__ == "__main__":
    main()
