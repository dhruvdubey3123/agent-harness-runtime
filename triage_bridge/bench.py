"""Crash every approved fix in the triage eval set and recover it two ways.

For each of the 100 labelled incidents: triage diagnoses it; if it reaches the human gate, a
reviewer approves; the harness starts applying the plan and the process is killed at a random
step, in one of three windows (before the tool runs, after it is marked dispatched but before it
runs, or after it ran but before the log recorded it). Then:

* restart: what a system without a run log does, run the whole approved plan again
* resume:  the harness reads its log; if it stops for review, a person checks the warehouse once

    python -m triage_bridge.bench --seed 1
"""
from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

from harness import Harness, RunStore, ScriptedModel, SimulatedCrash

from .runner import DATA, NeedsReview, TriageInHarness, plan_to_calls
from .warehouse import Warehouse

WINDOWS = ["before_tool", "during_tool", "after_tool"]


def crash_at(step: int, window: str):
    fired = {"done": False}

    def hook(event, idx, call):
        if not fired["done"] and idx == step and event == window:
            fired["done"] = True
            raise SimulatedCrash()
    return hook


def run(seed: int = 1) -> dict:
    rng = random.Random(seed)
    incidents = [json.loads(l) for l in (DATA / "incidents.jsonl").read_text().splitlines() if l.strip()]
    out = {"incidents": len(incidents), "reached_gate": 0, "escalated_by_triage": 0, "gate_matches_label": 0,
           "restart": {"runs_with_duplicate": 0, "duplicate_actions": 0, "runs_exactly_once": 0},
           "resume": {"runs_with_duplicate": 0, "duplicate_actions": 0, "finished_unattended": 0, "stopped_for_review": 0,
                      "finished": 0, "runs_exactly_once": 0}}
    for inc in incidents:
        with tempfile.TemporaryDirectory() as d:
            probe = TriageInHarness(Path(d) / "probe")
            view = probe.diagnose(inc)
            gated = view.get("waiting_for_human", False)
            out["gate_matches_label"] += gated == (inc["label"]["expected_action"] == "propose")
            if not gated:
                out["escalated_by_triage"] += 1
                continue
            out["reached_gate"] += 1
            calls = plan_to_calls(probe.dag, inc["id"], view["category"], view["plan"], view.get("proposal") or "")
            step, window = rng.randrange(len(calls)), rng.choice(WINDOWS)

            for strategy in ("restart", "resume"):
                wd = Path(d) / strategy
                t = TriageInHarness(wd, fault_hook=crash_at(step, window))
                t.diagnose(inc)
                try:
                    t.decide(inc["id"], True, "reviewer")
                except SimulatedCrash:
                    pass
                wh = Warehouse(wd / "warehouse.db")
                if strategy == "restart":
                    # no run log: the only option is to run the approved plan again from the top
                    Harness(wh.tools(), ScriptedModel(calls), RunStore()).start(
                        _agent(len(calls)), "re-apply from scratch", run_id="restart")
                else:
                    t2 = TriageInHarness(wd)
                    try:
                        v = t2.recover(inc["id"])
                        out["resume"]["finished_unattended"] += 1
                    except NeedsReview as e:
                        out["resume"]["stopped_for_review"] += 1
                        v = t2.resolve(inc["id"], t2.warehouse.took_effect(e.step["tool"], e.step["args"]))
                    out["resume"]["finished"] += v.get("outcome") == "resolved"
                dup = wh.duplicated_actions()
                out[strategy]["runs_exactly_once"] += exactly_once(wh, calls)
                out[strategy]["duplicate_actions"] += dup
                out[strategy]["runs_with_duplicate"] += dup > 0
    n = out["reached_gate"]
    for k in ("runs_with_duplicate", "runs_exactly_once"):
        out["restart"][k] = f'{out["restart"][k]}/{n}'
        out["resume"][k] = f'{out["resume"][k]}/{n}'
    for k in ("finished_unattended", "stopped_for_review", "finished"):
        out["resume"][k] = f'{out["resume"][k]}/{n}'
    out["gate_matches_label"] = f'{out["gate_matches_label"]}/{out["incidents"]}'
    return out


def exactly_once(wh: Warehouse, calls) -> bool:
    """Every action in the approved plan happened, and every non-repeatable one happened exactly once."""
    c = wh.counts()
    want_runs: dict[str, int] = {}
    for call in calls:
        if call.tool == "rerun_job":
            want_runs[call.args["node"]] = want_runs.get(call.args["node"], 0) + 1
    want_patch = {x.args["node"] for x in calls if x.tool == "patch_procedure"}
    want_refresh = {x.args["node"] for x in calls if x.tool == "refresh_table"}
    return (c["job_runs"] == want_runs and list(c["comments"].values()) == [1] and list(c["audit_lines"].values()) == [1]
            and set(c["patched"]) == want_patch and set(c["refreshed"]) == want_refresh)


def _agent(n: int):
    from harness import AgentSpec, Policy
    from .runner import TOOLS
    return AgentSpec("fixer", "Apply the approved plan exactly.", Policy(set(TOOLS)), max_steps=n + 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    res = run(a.seed)
    Path(__file__).with_name("bench_results.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
