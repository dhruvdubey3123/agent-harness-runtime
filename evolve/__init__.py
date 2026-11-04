"""A loop that tunes the harness against its own benchmark, and a gate that stops it cheating.

    from evolve import evolve, write
    ledger = evolve(generations=5)

The search space is in `knobs`, the gate in `guard`, the measurement in `scorer`, the loop in
`loop`. Nothing is merged automatically: a run writes a ledger and a person decides.
"""
from .knobs import FROZEN, SHIPPED, Knobs, neighbours
from .loop import Ledger, evolve, summary, write
from .scorer import Score, score

__all__ = ["FROZEN", "SHIPPED", "Knobs", "Ledger", "Score", "evolve", "neighbours", "score", "summary", "write"]
