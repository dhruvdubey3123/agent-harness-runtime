"""The part that has to be harder to fool than the part being optimised.

A search loop is only as trustworthy as the thing scoring it, and a benchmark scores what
happens to be in it. The 100 incidents never happen to contain a plan that runs the same job
twice, so a policy that is wrong exactly in that case would score perfectly and be accepted.

So the gate is two things:

1. invariants on the benchmark result (nothing duplicated, nothing dropped, the gate still
   agrees with the labels, every run still finishes),
2. four hand-written counterexamples below, built to break the specific shortcuts a search
   would otherwise be rewarded for taking.

A candidate has to pass all of it. `pause` and `unique_payload` do; `assume_done` and `retry`
are caught, which is how we know the probes do something.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from harness import AgentSpec, Harness, Policy, RunStore, ScriptedModel, SimulatedCrash, ToolCall
from triage_bridge.warehouse import Warehouse

from .knobs import NON_IDEMPOTENT, Knobs

TOOLS = {"patch_procedure", "rerun_job", "refresh_table", "comment_ticket", "append_audit"}

RERUN_X = ToolCall("rerun_job", {"node": "crm.daily_revenue"})
AUDIT = ToolCall("append_audit", {"line": "probe: applied fix"})


@dataclass(frozen=True)
class Probe:
    name: str
    why: str
    calls: tuple[ToolCall, ...]
    crash_idx: int
    window: str  # 'during_tool' = the tool never ran; 'after_tool' = it ran, the log missed it


PROBES = (
    Probe("repeated_action_never_ran",
          "the same job twice in one plan, killed before the second run: skipping it loses a run",
          (RERUN_X, RERUN_X, AUDIT), 1, "during_tool"),
    Probe("repeated_action_already_ran",
          "the same job twice, killed after the second run: repeating it makes a third",
          (RERUN_X, RERUN_X, AUDIT), 1, "after_tool"),
    Probe("unique_action_never_ran",
          "a one-off audit line killed before it was written: skipping it loses the line",
          (AUDIT,), 0, "during_tool"),
    Probe("unique_action_already_ran",
          "a one-off audit line killed after it was written: repeating it writes it twice",
          (AUDIT,), 0, "after_tool"),
)


def _wanted(calls: list[ToolCall]) -> dict[tuple[str, str], int]:
    """How many times the plan asks for each distinct effect. A plan may legitimately run the
    same job twice, so the probe compares against this rather than asking 'did anything happen
    more than once'; otherwise correct behaviour on a repeated step looks like a duplicate."""
    want: dict[tuple[str, str], int] = {}
    key = {"rerun_job": ("job_runs", "node"), "comment_ticket": ("comments", "case_id"),
           "append_audit": ("audit_lines", "line"), "patch_procedure": ("patched", "node"),
           "refresh_table": ("refreshed", "node")}
    for c in calls:
        table, field_ = key[c.tool]
        k = (table, str(c.args[field_]))
        want[k] = 1 if table in ("patched", "refreshed") else want.get(k, 0) + 1
    return want


def _actual(wh: Warehouse) -> dict[tuple[str, str], int]:
    c = wh.counts()
    got = {("job_runs", k): v for k, v in c["job_runs"].items()}
    got |= {("comments", k): v for k, v in c["comments"].items()}
    got |= {("audit_lines", k): v for k, v in c["audit_lines"].items()}
    got |= {("patched", k): 1 for k in c["patched"]}
    got |= {("refreshed", k): 1 for k in c["refreshed"]}
    return got


def compare(wh: Warehouse, calls: list[ToolCall]) -> tuple[int, int]:
    """(extra, missing) effects against what the approved plan actually asked for."""
    want, got = _wanted(calls), _actual(wh)
    keys = set(want) | set(got)
    extra = sum(max(0, got.get(k, 0) - want.get(k, 0)) for k in keys)
    missing = sum(max(0, want.get(k, 0) - got.get(k, 0)) for k in keys)
    return extra, missing


def _agent(n: int, retries: int) -> AgentSpec:
    return AgentSpec("fixer", "Apply the approved plan exactly.", Policy(set(TOOLS)), max_steps=n + 1, max_retries=retries)


def run_probe(probe: Probe, knobs: Knobs) -> str:
    """-> 'clean' | 'paused' | 'duplicated' | 'dropped'. Crash once, recover under `knobs`."""
    calls = list(probe.calls)
    with tempfile.TemporaryDirectory() as d:
        wh, db = Warehouse(Path(d) / "w.db"), Path(d) / "runs.db"
        fired = {"done": False}

        def hook(event, idx, call):
            if not fired["done"] and idx == probe.crash_idx and event == probe.window:
                fired["done"] = True
                raise SimulatedCrash()

        with RunStore(db) as first:
            try:
                Harness(wh.tools(), ScriptedModel(calls), first, fault_hook=hook).start(
                    _agent(len(calls), knobs.max_retries), "probe", run_id="probe")
            except SimulatedCrash:
                pass

        with RunStore(db) as store:  # a new process: only the log and the warehouse carried over
            Harness(wh.tools(), ScriptedModel(calls), store, recovery=knobs.policy(),
                    verifier=wh.took_effect).resume(_agent(len(calls), knobs.max_retries), "probe")
            paused = store.get_run("probe")["status"] == "needs_review"

        if paused:
            return "paused"  # correct, and costs a person
        extra, missing = compare(wh, calls)
        if extra:
            return "duplicated"
        if missing:
            return "dropped"
        return "clean"


def check_probes(knobs: Knobs) -> dict[str, str]:
    return {p.name: run_probe(p, knobs) for p in PROBES}


def probes_ok(results: dict[str, str]) -> bool:
    return all(r in ("clean", "paused") for r in results.values())


def check_invariants(metrics: dict, baseline: dict) -> dict[str, bool]:
    """Hard limits on the benchmark result. A candidate that breaks one is rejected however
    much it improved the thing we are optimising."""
    r, raw = metrics["resume"], metrics["raw"]
    gate = int(str(metrics["gate_matches_label"]).split("/")[0])
    base_gate = int(str(baseline["gate_matches_label"]).split("/")[0])
    return {
        "no_duplicated_actions": r["duplicate_actions"] == 0,
        "no_dropped_actions": r["dropped_actions"] == 0,
        "every_run_exactly_once": raw["resume"]["runs_exactly_once"] == raw["reached_gate"],
        "every_run_finished": raw["resume"]["finished"] == raw["reached_gate"],
        "gate_accuracy_not_reduced": gate >= base_gate,
        # Cheaper to pass than it looks, and it closes the loophole the other four leave open: a
        # policy that guesses an action already happened finishes unattended, which is what the
        # loop is optimising for, and leaves the report stale.
        "every_incident_resolved": raw["resume"].get("symptom_resolved") == raw["reached_gate"],
    }


def frozen_parts_intact(knobs: Knobs) -> dict[str, bool]:
    """The loop proposes a record, not code, so this is cheap to check: the knobs it returned
    still only name tools that are genuinely not safe to repeat, and nothing else moved."""
    try:
        knobs.validate()
    except ValueError:
        return {"knobs_within_declared_space": False}
    return {
        "knobs_within_declared_space": True,
        "only_non_idempotent_tools_touched": all(t in NON_IDEMPOTENT for t, _ in knobs.recovery),
    }
