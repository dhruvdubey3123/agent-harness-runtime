"""python -m evolve [--generations N] [--proposer rules|model] [--limit N] [--probes]"""
from __future__ import annotations

import argparse
import json

from . import guard, proposers
from .knobs import SHIPPED, Knobs
from .loop import evolve, summary, write


def main() -> None:
    ap = argparse.ArgumentParser(prog="evolve", description=__doc__)
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--proposer", default="rules", choices=("rules", "model"))
    ap.add_argument("--limit", type=int, default=None, help="score on the first N incidents only (faster)")
    ap.add_argument("--probes", action="store_true", help="just run the counterexamples against a few policies")
    a = ap.parse_args()

    if a.probes:
        for name, k in [("shipped (all pause)", SHIPPED),
                        ("unique_payload everywhere", Knobs(recovery=tuple(
                            (t, "unique_payload") for t in ("append_audit", "comment_ticket", "rerun_job")))),
                        ("assume_done on rerun_job", SHIPPED.with_recovery("rerun_job", "assume_done")),
                        ("retry on rerun_job", SHIPPED.with_recovery("rerun_job", "retry"))]:
            r = guard.check_probes(k)
            print(f"{'PASS' if guard.probes_ok(r) else 'FAIL'}  {name}")
            for probe, outcome in r.items():
                print(f"        {probe:32s} {outcome}")
        return

    ledger = evolve(generations=a.generations, seed=a.seed, proposer=proposers.get(a.proposer), limit=a.limit)
    out = write(ledger)
    print()
    print(summary(ledger))
    print(f"ledger: {out}/history.json")


if __name__ == "__main__":
    main()
