"""Score one candidate: run the probes, run the benchmark, apply the gate.

The objective is the share of crashed fixes the harness finishes without interrupting a
person. Every candidate is measured the same way on the same incidents and the same crash
points (the benchmark seeds its own RNG), so two scores are comparable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from triage_bridge import bench

from . import guard
from .knobs import Knobs


@dataclass
class Score:
    knobs: Knobs
    unattended: int = 0
    reached_gate: int = 0
    paused: int = 0
    duplicates: int = 0
    dropped: int = 0
    probes: dict[str, str] = field(default_factory=dict)
    invariants: dict[str, bool] = field(default_factory=dict)
    frozen: dict[str, bool] = field(default_factory=dict)
    paused_on: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0
    rejected_for: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict, repr=False)

    @property
    def ok(self) -> bool:
        return not self.rejected_for

    @property
    def objective(self) -> float:
        """Fraction finished without a human. Only meaningful for a candidate that passed."""
        return self.unattended / self.reached_gate if self.reached_gate else 0.0

    def line(self) -> str:
        verdict = "accept" if self.ok else "REJECT " + ", ".join(self.rejected_for)
        return (f"{self.unattended}/{self.reached_gate} unattended "
                f"({self.objective:.0%})  dup={self.duplicates} drop={self.dropped}  {verdict}")

    def to_dict(self) -> dict:
        return {"knobs": self.knobs.to_dict(), "unattended": self.unattended, "reached_gate": self.reached_gate,
                "paused": self.paused, "duplicates": self.duplicates, "dropped": self.dropped,
                "objective": round(self.objective, 4), "probes": self.probes, "invariants": self.invariants,
                "frozen": self.frozen, "paused_on": self.paused_on, "ok": self.ok,
                "rejected_for": self.rejected_for, "seconds": round(self.seconds, 1)}


def score(knobs: Knobs, baseline: dict | None = None, seed: int = 1, limit: int | None = None) -> Score:
    t0 = time.perf_counter()
    s = Score(knobs=knobs)

    s.frozen = guard.frozen_parts_intact(knobs)
    if not all(s.frozen.values()):
        s.rejected_for.append("outside the declared search space")
        s.seconds = time.perf_counter() - t0
        return s

    # Counterexamples first: they are fast, and a candidate that fails one is not worth
    # 28 seconds of benchmark.
    s.probes = guard.check_probes(knobs)
    if not guard.probes_ok(s.probes):
        s.rejected_for += [f"probe {n} -> {r}" for n, r in s.probes.items() if r not in ("clean", "paused")]
        s.seconds = time.perf_counter() - t0
        return s

    m = bench.run(seed=seed, recovery=knobs.policy(), min_confidence=knobs.min_confidence,
                  max_retries=knobs.max_retries, limit=limit)
    raw = m["raw"]
    s.reached_gate = raw["reached_gate"]
    s.unattended = raw["resume"]["finished_unattended"]
    s.paused = raw["resume"]["stopped_for_review"]
    s.duplicates = raw["resume"]["duplicate_actions"]
    s.dropped = raw["resume"]["dropped_actions"]
    s.paused_on = raw["resume"]["paused_on"]
    s.metrics = m
    s.invariants = guard.check_invariants(m, baseline or m)
    s.rejected_for += [f"invariant {n}" for n, passed in s.invariants.items() if not passed]
    s.seconds = time.perf_counter() - t0
    return s
