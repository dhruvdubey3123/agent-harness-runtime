"""Apply a SQL fix a person approved, through the harness, and keep it only if it verifies.

    proposal  ->  a person approves that exact SQL  ->  the fixer applies it inside the approval
              ->  the outcome is checked  ->  kept, or rolled back through the same run log

Where the proposal comes from is not this module's business: a model drafts it in
pipeline-triage-graph (triage/warehouse/fix.py), and a test can write one by hand. What this module
guarantees is what happens after:

  * The approval is for one plan, identified by a hash of the node and the SQL. A proposal whose SQL
    changed after it was approved has a different hash and is not applied.
  * The SQL is bound at the tool boundary. The fixer names the procedure and the runtime supplies
    the approved SQL; a fixer that writes its own is refused, in the run log.
  * Every call is in the crash-safe log, so a killed apply is resumed by the next process without
    recording the change twice.
  * After the apply, the warehouse is checked: the report the person complained about, and every
    other table. If either is wrong the prior definition is put back by a second harness run, so
    the rollback is journaled exactly like the fix.
"""
from __future__ import annotations

import difflib
import hashlib
from dataclasses import dataclass

from harness import AgentSpec, Envelope, Harness, Policy, RecoveryPolicy, RunStore, ScriptedModel
from harness.types import ToolCall

from .sql_warehouse import SqlWarehouse

TOOLS = ("patch_procedure", "run_cycle", "record_change")
KEY_FIELDS = {"patch_procedure": ("node", "sql"), "run_cycle": (), "record_change": ("case_id",)}
BOUND_FIELDS = {"patch_procedure": ("sql",)}        # content, fixed by the approval

FIXER = ("You are the fixer for a data warehouse. A person has read a proposed SQL fix and approved it. Carry it out "
         "one tool call at a time: deploy the approved SQL with patch_procedure (name the node and leave sql out, the "
         "approved SQL is supplied for you), rebuild with run_cycle, then record_change with one sentence in your own "
         "words saying what was changed and why. The approved actions are the limit: the runtime refuses anything "
         "else and writes the refusal to the run log. Apply each approved action once, then finish.")
REPLAY = "Apply the approved plan exactly."


class NotApproved(Exception):
    pass


class NeedsReview(Exception):
    def __init__(self, run_id: str, step: dict):
        super().__init__(f"{run_id}: step {step['idx']} ({step['tool']}) was interrupted and is not safe to repeat")
        self.run_id, self.step = run_id, step


@dataclass(frozen=True)
class Proposal:
    case_id: str
    symptom_node: str
    node: str
    sql: str
    reason: str = ""

    @property
    def plan_id(self) -> str:
        """What a person approves. Any change to the node or the SQL is a different plan."""
        return hashlib.sha256(f"{self.node}\n{self.sql.strip().rstrip(';').strip()}".encode()).hexdigest()[:12]

    def calls(self) -> list[ToolCall]:
        return [ToolCall("patch_procedure", {"node": self.node, "sql": self.sql.strip().rstrip(";").strip()}),
                ToolCall("run_cycle", {}),
                ToolCall("record_change", {"case_id": self.case_id, "body": f"deployed approved fix to {self.node}: {self.reason}"})]


def briefing(wh: SqlWarehouse, p: Proposal, rehearsal: str | None = None) -> str:
    """What a person reads before deciding: the node, the change as a diff, and what the rehearsal said."""
    before = wh.deployed(p.node).replace(" from ", "\nfrom ").replace(" where ", "\nwhere ").replace(" left join ", "\nleft join ").replace(" join ", "\njoin ")
    after = p.sql.replace(" from ", "\nfrom ").replace(" where ", "\nwhere ").replace(" left join ", "\nleft join ").replace(" join ", "\njoin ")
    diff = "\n".join(difflib.unified_diff(before.splitlines(), after.splitlines(), "deployed", "proposed", lineterm="", n=1))
    return (f"Case {p.case_id}: {p.symptom_node} looks wrong\n"
            f"Proposed fix: replace the SQL of {p.node}\n"
            f"Why: {p.reason or '(no reason given)'}\n"
            f"Rehearsal: {rehearsal or 'not run'}\n\n{diff or '(no change)'}\n\n"
            f"Plan id: {p.plan_id}")


def _run(wh: SqlWarehouse, store: RunStore, run_id: str, calls: list[ToolCall], model, instructions: str,
         task: str, fault_hook=None, recovery: RecoveryPolicy | None = None, extra_steps: int = 0) -> dict:
    envelope = Envelope.from_calls(calls, KEY_FIELDS, BOUND_FIELDS)
    agent = AgentSpec("fixer", instructions, Policy(set(TOOLS), envelope=envelope), max_steps=len(calls) + extra_steps)
    h = Harness(wh.tools(), model, store, fault_hook=fault_hook, recovery=recovery or RecoveryPolicy(),
                verifier=wh.took_effect)
    if store.get_run(run_id) is None:
        h.start(agent, task, task_kind="sql_fix", run_id=run_id)
    else:
        h.resume(agent, run_id)
    run, steps = store.get_run(run_id), store.steps(run_id)
    if run["status"] == "needs_review":
        raise NeedsReview(run_id, steps[-1])
    return {"run_id": run_id, "status": run["status"], "answer": run["answer"],
            "steps": [f'{s["tool"]} -> {s["status"]}' + (f' ({s["error"][:90]})' if s["error"] else "") for s in steps]}


def apply(wh: SqlWarehouse, store: RunStore, p: Proposal, approved_plan_id: str | None, model=None,
          fault_hook=None, recovery: RecoveryPolicy | None = None) -> dict:
    """Apply the proposal if, and only if, this is the plan a person approved. `model` is the fixer;
    left out, the approved calls are replayed with no model."""
    if approved_plan_id != p.plan_id:
        raise NotApproved(f"plan {p.plan_id} is not the plan that was approved ({approved_plan_id or 'nothing was'})")
    calls = p.calls()
    replaying = model is None
    task = (f"Apply the approved fix for {p.case_id}. Approved actions: "
            f"{Envelope.from_calls(calls, KEY_FIELDS, BOUND_FIELDS).describe()}")
    return _run(wh, store, f"sqlfix-{p.case_id}", calls, ScriptedModel(calls) if replaying else model,
                REPLAY if replaying else FIXER, task, fault_hook, recovery, extra_steps=1 if replaying else 6)


def verify_or_roll_back(wh: SqlWarehouse, store: RunStore, p: Proposal, fault_hook=None) -> dict:
    """Two answers, never added: is the report right, and is everything else. Both, or it is undone."""
    out = wh.outcome(p.symptom_node, allowed=(p.node,))
    out["kept"] = out["functional"] and out["integrity"]
    if not out["kept"]:
        calls = wh.rollback_calls(p.case_id, p.node)
        out["rollback"] = _run(wh, store, f"rollback-{p.case_id}", calls, ScriptedModel(calls), REPLAY,
                               f"roll back the fix for {p.case_id}", fault_hook, extra_steps=1)
    return out
