"""Triage diagnoses, a person approves, the harness applies.

The triage graph already has an extension point for "doing the fix": its Executor. Here the
Executor is the harness. When the graph reaches its apply step, the approved plan becomes a
durable harness run over the warehouse tools. Both halves keep their state in SQLite (the graph
through LangGraph's SqliteSaver, the harness through its run log), so a process can die at any
point and a new process picks up where it stopped.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from triage.dag import PipelineDAG
from triage.knowledge import CaseLibrary
from triage.service import TriageService

from harness import AgentSpec, Harness, Policy, RunStore, ScriptedModel, ToolCall

from .warehouse import Warehouse

DATA = Path(__file__).resolve().parent / "data"
TOOLS = {"patch_procedure", "rerun_job", "refresh_table", "comment_ticket", "append_audit"}


class NeedsReview(Exception):
    """The harness stopped on a step that may already have happened and is not safe to repeat."""

    def __init__(self, run_id: str, step: dict):
        self.run_id, self.step = run_id, step
        super().__init__(f"{run_id}: step {step['idx']} {step['tool']}({step['args']}) needs a person to check it")


def plan_to_calls(dag: PipelineDAG, case_id: str, category: str, plan: list[str], proposal: str) -> list[ToolCall]:
    """Turn triage's approved plan ("fix X", "refresh Y", ...) into concrete warehouse actions."""
    calls: list[ToolCall] = []
    for step in plan:
        verb, node = step.split(" ", 1)
        kind = dag.nodes[node].kind
        if verb == "fix":
            if kind == "procedure":
                calls.append(ToolCall("patch_procedure", {"node": node, "change": category}))
            calls.append(ToolCall("rerun_job", {"node": node}))
        elif kind in ("load", "procedure"):
            calls.append(ToolCall("rerun_job", {"node": node}))
        else:
            calls.append(ToolCall("refresh_table", {"node": node}))
    calls.append(ToolCall("comment_ticket", {"case_id": case_id, "body": f"Resolved: {proposal}"}))
    calls.append(ToolCall("append_audit", {"line": f"{case_id}: applied {category} fix, {len(plan)} plan steps"}))
    return calls


class HarnessExecutor:
    """Triage's Executor protocol, implemented by the harness. Only the approved plan can run:
    the 'model' here is the plan itself, so the fixer cannot improvise steps nobody reviewed."""

    def __init__(self, runner: "TriageInHarness"):
        self.r = runner
        self.case_id: str | None = None

    def apply(self, plan: list[str]) -> list[str]:
        r, cid = self.r, self.case_id
        view = r.service.view(cid)
        calls = plan_to_calls(r.dag, cid, view["category"], plan, view.get("proposal") or "")
        agent = AgentSpec("fixer", "Apply the approved plan exactly.", Policy(set(TOOLS)), max_steps=len(calls) + 1)
        h = Harness(r.warehouse.tools(), ScriptedModel(calls), r.store, fault_hook=r.fault_hook)
        rid = f"apply-{cid}"
        if r.store.get_run(rid) is None:
            h.start(agent, f"apply approved fix for {cid}", task_kind="triage_apply", run_id=rid)
        else:
            h.resume(agent, rid)
        run = r.store.get_run(rid)
        if run["status"] == "needs_review":
            raise NeedsReview(rid, r.store.steps(rid)[-1])
        return [f'{s["tool"]}({s["args"].get("node") or s["args"].get("case_id") or ""}) -> {s["status"]}' for s in r.store.steps(rid)]


class TriageInHarness:
    def __init__(self, workdir: str | Path, fault_hook=None, min_confidence: float = 0.5):
        self.dir = Path(workdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dag = PipelineDAG.from_yaml(DATA / "pipeline.yaml")
        self.warehouse = Warehouse(self.dir / "warehouse.db")
        self.store = RunStore(self.dir / "harness_runs.db")
        self.fault_hook = fault_hook
        self.executor = HarnessExecutor(self)
        saver = SqliteSaver(sqlite3.connect(str(self.dir / "triage_graph.db"), check_same_thread=False))
        self.service = TriageService(self.dag, CaseLibrary.load(DATA / "case_library.yaml"), executor=self.executor,
                                     checkpointer=saver, min_confidence=min_confidence)

    def _cfg(self, cid: str) -> dict:
        return {"configurable": {"thread_id": cid}}

    # 1. triage proposes, and the graph pauses before any change
    def diagnose(self, incident: dict) -> dict:
        _, view = self.service.open(incident)
        return view

    # 2. a person decides; on approve, the harness applies the plan
    def decide(self, case_id: str, approve: bool, reviewer: str = "on-call") -> dict:
        self.executor.case_id = case_id
        return self.service.decide(case_id, approve, reviewer)

    # 3. after a crash: finish the apply step from both logs, never repeating a finished action
    def recover(self, case_id: str) -> dict:
        self.executor.case_id = case_id
        snap = self.service.app.get_state(self._cfg(case_id))
        if "apply" in (snap.next or ()):
            self.service.app.invoke(None, self._cfg(case_id))
        return self.service.view(case_id)

    # 4. a person checked the interrupted step: record what really happened, then carry on
    def resolve(self, case_id: str, took_effect: bool) -> dict:
        rid = f"apply-{case_id}"
        last = self.store.steps(rid)[-1]
        self.store.resolve_review(rid, last["idx"], took_effect)
        return self.recover(case_id)

    def apply_status(self, case_id: str) -> dict | None:
        rid = f"apply-{case_id}"
        run = self.store.get_run(rid)
        return None if run is None else {**run, "steps": self.store.steps(rid)}
