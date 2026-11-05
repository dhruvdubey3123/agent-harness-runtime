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

from harness import (AgentSpec, Envelope, Harness, OpenAICompatModel, Policy, RecoveryPolicy, RunStore, ScriptedModel,
                     ToolCall)

from .warehouse import Warehouse

DATA = Path(__file__).resolve().parent / "data"
TOOLS = {"patch_procedure", "rerun_job", "refresh_table", "comment_ticket", "append_audit"}

# What has to match the approval exactly: which node, which case. The text of a resolution comment
# and of an audit line is left to whoever is applying the fix, which is the part a model is for.
KEY_FIELDS = {"patch_procedure": ("node",), "rerun_job": ("node",), "refresh_table": ("node",),
              "comment_ticket": ("case_id",), "append_audit": ()}

REPLAY = "Apply the approved plan exactly."
FIXER = ("You are the fixer for a data pipeline. A person has already read the diagnosis and approved a plan; "
         "your job is to carry it out against the warehouse, one tool call at a time. The order is yours, and "
         "any wording is yours. The approved actions are the limit: the runtime refuses anything else and writes "
         "the refusal to the run log, so there is nothing to gain by trying a variation of a refused call. Apply "
         "each approved action once, then finish with one sentence saying what you did.")


class Incomplete(Exception):
    """The fixer said it was finished while approved actions had not happened."""


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


def describe_task(case_id: str, view: dict, envelope: Envelope) -> str:
    """What the fixer is told. The approved actions are in the prompt and in the sandbox, so a model
    that ignores the prompt changes nothing about what it is able to do."""
    lines = [f"Incident {case_id} on the data pipeline.",
             f"Category: {view.get('category')}",
             f"Diagnosis a person approved: {view.get('proposal') or '(none recorded)'}",
             "", "Approved actions:"]
    lines += [f"  {i + 1}. {a}" for i, a in enumerate(envelope.actions)]
    lines += ["", "Arguments shown above are fixed. Any other argument (a comment body, an audit line) is yours "
                  "to write, and should name the case and what was fixed."]
    return "\n".join(lines)


class HarnessExecutor:
    """Triage's Executor protocol, implemented by the harness.

    Whoever applies the fix, the approval is enforced by the sandbox rather than by trust. With the
    scripted fixer the plan is replayed call for call. With a real model the plan becomes an
    envelope: the model decides how to work and the runtime refuses anything a person did not
    approve. Both paths leave the same run log.
    """

    def __init__(self, runner: "TriageInHarness"):
        self.r = runner
        self.case_id: str | None = None

    def apply(self, plan: list[str]) -> list[str]:
        r, cid = self.r, self.case_id
        view = r.service.view(cid)
        calls = plan_to_calls(r.dag, cid, view["category"], plan, view.get("proposal") or "")
        envelope = Envelope.from_calls(calls, KEY_FIELDS)
        replaying = r.fixer == "scripted"
        model = ScriptedModel(calls) if replaying else r.fixer_model()
        agent = AgentSpec("fixer", REPLAY if replaying else FIXER,
                          Policy(set(TOOLS), envelope=envelope),
                          max_steps=len(calls) + (1 if replaying else 5), max_retries=r.max_retries)
        h = Harness(r.warehouse.tools(), model, r.store, fault_hook=r.fault_hook,
                    recovery=r.recovery, verifier=r.warehouse.took_effect)
        rid = f"apply-{cid}"
        if r.store.get_run(rid) is None:
            task = f"apply approved fix for {cid}" if replaying else describe_task(cid, view, envelope)
            h.start(agent, task, task_kind="triage_apply", run_id=rid)
        else:
            h.resume(agent, rid)
        run = r.store.get_run(rid)
        if run["status"] == "needs_review":
            raise NeedsReview(rid, r.store.steps(rid)[-1])
        if run["status"] == "failed":
            raise Incomplete(f"{rid}: {run['answer']}")
        return [f'{s["tool"]}({s["args"].get("node") or s["args"].get("case_id") or ""}) -> {s["status"]}' for s in r.store.steps(rid)]


class TriageInHarness:
    def __init__(self, workdir: str | Path, fault_hook=None, min_confidence: float = 0.5,
                 recovery: RecoveryPolicy | None = None, max_retries: int = 2, fixer="scripted", communicator=None):
        """fixer: "scripted" replays the approved plan, "model" hands it to a local open-weight
        model through an OpenAI-compatible server, and any object with next_action is used as is.
        communicator: a chat model that writes the briefing the person at the gate reads, from the
        tracer's findings, checked against them (pipeline-triage-graph's triage/communicator.py)."""
        self.dir = Path(workdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dag = PipelineDAG.from_yaml(DATA / "pipeline.yaml")
        self.warehouse = Warehouse(self.dir / "warehouse.db")
        self.store = RunStore(self.dir / "harness_runs.db")
        self.fault_hook = fault_hook
        self.recovery = recovery or RecoveryPolicy()
        self.max_retries = max_retries
        self.fixer = fixer
        self.executor = HarnessExecutor(self)
        self._graph_db = sqlite3.connect(str(self.dir / "triage_graph.db"), check_same_thread=False)
        self.service = TriageService(self.dag, CaseLibrary.load(DATA / "case_library.yaml"), executor=self.executor,
                                     checkpointer=SqliteSaver(self._graph_db), min_confidence=min_confidence,
                                     communicator=communicator)

    def close(self) -> None:
        """Both SQLite files, released. Needed wherever the working directory is deleted after,
        since Windows refuses to delete a file that is still open."""
        self._graph_db.close()
        self.store.close()

    def __enter__(self) -> TriageInHarness:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def fixer_model(self):
        return OpenAICompatModel() if self.fixer == "model" else self.fixer

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
