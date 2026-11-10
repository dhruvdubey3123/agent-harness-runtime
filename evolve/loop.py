"""The loop: propose a change, measure it, keep it only if it is better and still safe.

    shipped config ──► propose ──► score ──► passed the gate and better? ──► new best
                          ▲                            │ no
                          └──────── next candidate ◄───┘

Nothing here is merged automatically. The loop writes what it found to `evolve/results/`
and a person reads the ledger and decides. The honest summary of a run is in two numbers:
how much the objective moved, and how many candidates the gate threw out.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .knobs import FROZEN, SHIPPED, Knobs
from .proposers import Proposer, RuleProposer
from .scorer import Score, score

RESULTS = Path(__file__).resolve().parent / "results"


@dataclass
class Ledger:
    baseline: Score
    best: Score
    history: list[Score] = field(default_factory=list)
    generations: int = 0
    proposer: str = "rules"

    @property
    def accepted(self) -> list[Score]:
        return [s for s in self.history if s.ok]

    @property
    def rejected(self) -> list[Score]:
        return [s for s in self.history if not s.ok]

    def to_dict(self) -> dict:
        return {
            "proposer": self.proposer,
            "generations": self.generations,
            "candidates_scored": len(self.history),
            "candidates_rejected_by_the_gate": len(self.rejected),
            "frozen_during_the_search": list(FROZEN),
            "baseline": self.baseline.to_dict(),
            "best": self.best.to_dict(),
            "moved": {"unattended": f"{self.baseline.unattended} -> {self.best.unattended} of {self.best.reached_gate}",
                      "objective": f"{self.baseline.objective:.0%} -> {self.best.objective:.0%}",
                      "duplicated_actions": f"{self.baseline.duplicates} -> {self.best.duplicates}",
                      "dropped_actions": f"{self.baseline.dropped} -> {self.best.dropped}"},
            "history": [s.to_dict() for s in self.history],
        }


def evolve(generations: int = 5, seed: int = 1, proposer: Proposer | None = None,
           limit: int | None = None, start: Knobs = SHIPPED, verbose: bool = True) -> Ledger:
    proposer = proposer or RuleProposer()
    say = print if verbose else (lambda *a, **k: None)

    say(f"baseline  {start.label()}")
    baseline = score(start, seed=seed, limit=limit)
    say(f"          {baseline.line()}  ({baseline.seconds:.0f}s)")
    if not baseline.ok:
        raise RuntimeError(f"the shipped configuration does not pass its own gate: {baseline.rejected_for}")

    best, tried = baseline, {start}
    ledger = Ledger(baseline=baseline, best=best, history=[baseline], proposer=proposer.name)

    for gen in range(1, generations + 1):
        cands = proposer.propose(best, tried)
        if not cands:
            say("no candidates left to try")
            break
        cand = cands[0]
        tried.add(cand)
        say(f"gen {gen}    {cand.label()}")
        s = score(cand, baseline=baseline.metrics, seed=seed, limit=limit)
        ledger.history.append(s)
        say(f"          {s.line()}  ({s.seconds:.0f}s)")
        if s.ok and s.objective > best.objective:
            best, ledger.best = s, s
            say(f"          kept: {baseline.objective:.0%} -> {s.objective:.0%} unattended")
        ledger.generations = gen

    return ledger


def write(ledger: Ledger, out: Path = RESULTS) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    (out / "history.json").write_text(json.dumps(ledger.to_dict(), indent=2) + "\n")
    (out / "summary.md").write_text(summary(ledger))
    return out


def summary(ledger: Ledger) -> str:
    b, best = ledger.baseline, ledger.best
    rows = ["# Recovery policy search", "",
            f"Proposer: `{ledger.proposer}`. {len(ledger.history)} configurations measured on the same "
            f"{best.reached_gate} crashed fixes; {len(ledger.rejected)} rejected by the gate.", "",
            "| configuration | unattended | duplicated | dropped | verdict |", "|---|---|---|---|---|"]
    for s in ledger.history:
        verdict = "kept" if s is best else ("passed" if s.ok else "rejected: " + "; ".join(s.rejected_for))
        rows.append(f"| `{s.knobs.label()}` | {s.unattended}/{s.reached_gate} | {s.duplicates} | {s.dropped} | {verdict} |")
    rows += ["",
             f"**Result.** Unattended recovery went from {b.unattended}/{b.reached_gate} to "
             f"{best.unattended}/{best.reached_gate} ({b.objective:.0%} to {best.objective:.0%}). "
             f"Duplicated actions stayed at {best.duplicates} and dropped actions at {best.dropped}; "
             "a candidate that changed either was thrown away.", "",
             "**Frozen during the search:**"] + [f"- {f}" for f in FROZEN]
    return "\n".join(rows) + "\n"
