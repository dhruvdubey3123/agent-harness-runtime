"""Crash-recovery benchmark: restart-from-scratch vs harness resume.

Each trial runs an 8-step file task that mixes idempotent writes with non-idempotent appends
to an audit log, kills the process at a random step in one of three windows (before the tool
is dispatched, after dispatch but before it takes effect, or after it took effect but before
the result was logged), then recovers one of two ways:

  restart  start the task again from step 0 with no memory of the first attempt
  resume   reopen the same run from the SQLite log with Harness.resume

and checks what ended up on disk. The model is scripted, so both strategies make the same
decisions; the only variable is how recovery works.

    python -m bench.crash_bench --trials 200 --seed 1
"""
from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

from harness import AgentSpec, Harness, Policy, RunStore, ScriptedModel, SimulatedCrash, ToolCall, workspace_tools

POLICY = Policy({"write_file", "append_line", "read_file"})


def make_script(rng: random.Random, n: int = 8) -> list[ToolCall]:
    out = []
    for i in range(n):
        if rng.random() < 0.4:
            out.append(ToolCall("append_line", {"path": "audit.log", "line": f"event {i}"}))
        else:
            out.append(ToolCall("write_file", {"path": f"out/file{i}.txt", "content": f"content {i}"}))
    return out


def counting_registry(ws: Path, counter: dict):
    reg = workspace_tools(ws)
    for name in ("write_file", "append_line"):
        t = reg.get(name)
        orig = t.fn

        def wrapped(*a, _orig=orig, _n=name, **k):
            key = _n + json.dumps(k, sort_keys=True)
            counter[key] = counter.get(key, 0) + 1
            return _orig(*a, **k)
        t.fn = wrapped
    return reg


def crash_hook(event: str, step: int):
    state = {"fired": False}

    def hook(ev, idx, call):
        if not state["fired"] and ev == event and idx == step:
            state["fired"] = True
            raise SimulatedCrash()
    return hook


def trial(rng: random.Random, strategy: str) -> dict:
    script = make_script(rng)
    step = rng.randrange(len(script))
    window = rng.choice(["before_tool", "during_tool", "after_tool"])
    expected_lines = [c.args["line"] for c in script if c.tool == "append_line"]
    with tempfile.TemporaryDirectory() as d:
        ws, db, counter = Path(d) / "ws", Path(d) / "runs.db", {}
        agent = AgentSpec("bench", "run the script", POLICY)
        with RunStore(db) as first:   # closed before the directory goes: Windows will not delete an open file
            h1 = Harness(counting_registry(ws, counter), ScriptedModel(script), first,
                         fault_hook=crash_hook(window, step))
            try:
                h1.start(agent, "task", run_id="t")
            except SimulatedCrash:
                pass
        status = "done"
        if strategy == "restart":
            with RunStore() as fresh:
                Harness(counting_registry(ws, counter), ScriptedModel(script), fresh).start(agent, "task")
        else:
            with RunStore(db) as second:
                h2 = Harness(counting_registry(ws, counter), ScriptedModel(script), second)
                h2.resume(agent, "t")
                status = h2.store.get_run("t")["status"]
        audit = ws / "audit.log"
        lines = audit.read_text().splitlines() if audit.exists() else []
        repeated = sum(c - 1 for c in counter.values() if c > 1)
    return {
        "repeated_executions": repeated,
        "duplicated_side_effect": len(lines) > len(expected_lines),
        "halted_for_review": status == "needs_review",
        "correct_unattended": status == "done" and lines == expected_lines,
        "window": window,
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "trials": n,
        "side_effecting_calls_repeated": sum(r["repeated_executions"] for r in rows),
        "runs_with_duplicated_side_effect": f"{sum(r['duplicated_side_effect'] for r in rows)}/{n}",
        "runs_halted_for_human_review": f"{sum(r['halted_for_review'] for r in rows)}/{n}",
        "runs_correct_without_human": f"{sum(r['correct_unattended'] for r in rows)}/{n}",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="bench/results.json")
    a = ap.parse_args()
    res = {}
    for strategy in ("restart", "resume"):
        rng = random.Random(a.seed)  # same scripts and crash points for both strategies
        res[strategy] = summarize([trial(rng, strategy) for _ in range(a.trials)])
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
