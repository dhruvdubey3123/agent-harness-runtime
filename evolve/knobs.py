"""What the loop is allowed to change, and what it is not.

The search space is a small typed record, not source code. That is the point: a loop that
can rewrite anything can also rewrite the thing that judges it, and then its score stops
meaning anything. Here the loop may only move these knobs, and `FROZEN` lists what it may
never touch. `Knobs.validate` is the enforcement, and `evolve.guard` re-checks the frozen
parts after every candidate in case a proposer returns something malformed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Iterator

from harness import STRATEGIES, RecoveryPolicy

# Tools the fixer may use, with whether repeating one is safe. Not a knob: this is a fact
# about the warehouse, and a loop that could edit it would simply relabel its own mistakes.
NON_IDEMPOTENT = ("rerun_job", "comment_ticket", "append_audit")

RETRIES = (0, 1, 2, 3, 4)
CONFIDENCE = (0.3, 0.4, 0.5, 0.6, 0.7)

FROZEN = (
    "the tool allowlist each agent gets (harness.Policy)",
    "which tools are marked idempotent (triage_bridge.warehouse)",
    "the human approval gate and its position in the graph",
    "the benchmark, its incidents and their labels",
    "the safety invariants and the counterexample probes in evolve.guard",
)


@dataclass(frozen=True)
class Knobs:
    """One candidate configuration. Frozen and hashable so the loop can dedupe what it tried."""

    recovery: tuple[tuple[str, str], ...] = ()
    max_retries: int = 2
    min_confidence: float = 0.5

    # ------------------------------------------------------------------ shape
    @property
    def recovery_map(self) -> dict[str, str]:
        return dict(self.recovery)

    def policy(self) -> RecoveryPolicy:
        return RecoveryPolicy(per_tool=self.recovery_map)

    def validate(self) -> None:
        for tool, strategy in self.recovery:
            if tool not in NON_IDEMPOTENT:
                raise ValueError(f"{tool} is not a non-idempotent tool; recovery strategies do not apply to it")
            if strategy not in STRATEGIES:
                raise ValueError(f"unknown strategy {strategy!r}")
        if self.max_retries not in RETRIES:
            raise ValueError(f"max_retries must be one of {RETRIES}")
        if self.min_confidence not in CONFIDENCE:
            raise ValueError(f"min_confidence must be one of {CONFIDENCE}")

    def with_recovery(self, tool: str, strategy: str) -> "Knobs":
        m = self.recovery_map
        m[tool] = strategy
        # 'pause' is the default, so drop it: two records that mean the same policy must be
        # the same value, or the loop re-measures configurations it has already tried.
        return replace(self, recovery=tuple(sorted((t, s) for t, s in m.items() if s != "pause")))

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return {"recovery": self.recovery_map, "max_retries": self.max_retries, "min_confidence": self.min_confidence}

    @classmethod
    def from_dict(cls, d: dict) -> "Knobs":
        k = cls(recovery=tuple(sorted((t, s) for t, s in (d.get("recovery") or {}).items() if s != "pause")),
                max_retries=int(d.get("max_retries", 2)),
                min_confidence=float(d.get("min_confidence", 0.5)))
        k.validate()
        return k

    def label(self) -> str:
        r = ", ".join(f"{t}={s}" for t, s in self.recovery) or "all tools pause"
        return f"[{r}] retries={self.max_retries} gate>={self.min_confidence}"

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


SHIPPED = Knobs()
"""The configuration in main before any search: every non-repeatable tool stops for a person."""


def neighbours(k: Knobs) -> Iterator[Knobs]:
    """Every candidate one knob away. The whole space the loop may move through in one step."""
    for tool in NON_IDEMPOTENT:
        for strategy in STRATEGIES:
            if k.recovery_map.get(tool, "pause") != strategy:
                yield k.with_recovery(tool, strategy)
    for r in RETRIES:
        if r != k.max_retries:
            yield replace(k, max_retries=r)
    for c in CONFIDENCE:
        if c != k.min_confidence:
            yield replace(k, min_confidence=c)
