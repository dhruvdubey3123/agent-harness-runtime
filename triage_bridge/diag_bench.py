"""Is the model any good at the diagnosis? Scored against the labels, next to the rule-based path.

The incidents carry labels: which node had to be fixed, which upstream node caused it, and the
category. The triage graph finds the root cause by walking the pipeline structure and matches the
labels on all 100, which is the number a model has to be measured against rather than compared to
a vague sense of "it worked".

    python -m triage_bridge.diag_bench --limit 20        # needs a local model server

Nothing is applied here and the fixer never runs: this scores the first agent only, so a wrong
answer costs a wrong row in the table and nothing else.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from triage.dag import PipelineDAG
from triage.knowledge import CaseLibrary
from triage.service import TriageService

from harness import OpenAICompatModel

from .orchestrate import TwoAgentTriage
from .runner import DATA

OUT = Path(__file__).resolve().parent / "diag_bench_results.json"


def incidents(limit: int | None = None, only: list[str] | None = None, skip: int = 0) -> list[dict]:
    rows = [json.loads(line) for line in (DATA / "incidents.jsonl").read_text().splitlines()]
    if only:
        rows = [r for r in rows if r["id"] in only]
    rows = rows[skip:]
    return rows[:limit] if limit else rows


def score(label: dict, target: str | None, category: str | None) -> dict:
    return {"root_cause_ok": bool(target) and target in label["fix_targets"],
            "category_ok": category == label["category"]}


def rules(rows: list[dict]) -> dict:
    """The baseline: the same incidents through the deterministic graph, which applies nothing
    because it is compiled to interrupt before the gate."""
    dag = PipelineDAG.from_yaml(DATA / "pipeline.yaml")
    svc = TriageService(dag, CaseLibrary.load(DATA / "case_library.yaml"))
    t0, out = time.perf_counter(), []
    for r in rows:
        _, v = svc.open(r)
        out.append({"id": r["id"], "fix_target": v.get("fix_target"), "category": v.get("category"),
                    **score(r["label"], v.get("fix_target"), v.get("category"))})
    return {"cases": out, "seconds": round(time.perf_counter() - t0, 1)}


def model_path(rows: list[dict], model, workdir: Path, max_steps: int) -> dict:
    t0, out = time.perf_counter(), []
    for r in rows:
        with TwoAgentTriage(workdir / r["id"], model=model, max_steps=max_steps) as t:
            try:
                t.diagnose(r)
            except Exception as e:                  # a model that stalls should cost one row, not the run
                out.append({"id": r["id"], "error": f"{type(e).__name__}: {e}",
                            "root_cause_ok": False, "category_ok": False})
                continue
            d, steps = t.diagnosis(r["id"]), t.store.steps(f"diag-{r['id']}")
            out.append({"id": r["id"], "fix_target": d and d.fix_target, "category": d and d.category,
                        "cause": d and d.cause, "submitted": d is not None, "tool_calls": len(steps),
                        "refused": sum(1 for s in steps if s.get("error")),
                        "review_issues": t.review(r["id"]),
                        **score(r["label"], d and d.fix_target, d and d.category)})
    return {"cases": out, "seconds": round(time.perf_counter() - t0, 1)}


def tally(cases: list[dict], key: str) -> str:
    return f"{sum(1 for c in cases if c.get(key))}/{len(cases)}"


def run(model=None, limit: int | None = None, only: list[str] | None = None, workdir: Path | None = None,
        max_steps: int = 14, skip: int = 0) -> dict:
    rows = incidents(limit, only, skip)
    model = model or OpenAICompatModel()
    workdir = workdir or Path(__file__).resolve().parent / "diag_bench_runs"
    m, r = model_path(rows, model, workdir, max_steps), rules(rows)
    return {"model": getattr(model, "model", "?"), "incidents": len(rows),
            "model_diagnoser": {"root cause correct": tally(m["cases"], "root_cause_ok"),
                                "category correct": tally(m["cases"], "category_ok"),
                                "submitted a diagnosis": tally(m["cases"], "submitted"),
                                "mean tool calls": round(sum(c.get("tool_calls", 0) for c in m["cases"])
                                                         / max(len(m["cases"]), 1), 1),
                                "seconds": m["seconds"]},
            "rule_based_tracer": {"root cause correct": tally(r["cases"], "root_cause_ok"),
                                  "category correct": tally(r["cases"], "category_ok"),
                                  "seconds": r["seconds"]},
            "raw": {"model": m["cases"], "rules": r["cases"]}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=None, help="first N incidents (default: all 100)")
    p.add_argument("--skip", type=int, default=0, help="skip the first N incidents, for a held-out sample")
    p.add_argument("--only", nargs="*", default=None, help="case ids, for a single run")
    p.add_argument("--max-steps", type=int, default=14)
    a = p.parse_args()
    res = run(limit=a.limit, only=a.only, max_steps=a.max_steps, skip=a.skip)
    print(f"\n{res['model']} on {res['incidents']} labelled incidents\n")
    keys = ["root cause correct", "category correct", "submitted a diagnosis", "mean tool calls", "seconds"]
    print(f"{'':24}{'model diagnoser':>18}{'rule-based tracer':>20}")
    for k in keys:
        print(f"{k:24}{str(res['model_diagnoser'].get(k, '')):>18}{str(res['rule_based_tracer'].get(k, 'n/a')):>20}")
    wrong = [c["id"] for c in res["raw"]["model"] if not c["root_cause_ok"]]
    if wrong:
        print(f"\nroot cause missed on: {', '.join(wrong)}")
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nFull results: {OUT}")


if __name__ == "__main__":
    main()
