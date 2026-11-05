"""Two agents and a person, orchestrated by the harness rather than by a framework.

    diagnose (model, read-only tools)  ->  review  ->  a person approves  ->  apply (model, inside
    the approved envelope)

Both agents are ordinary harness runs in one SQLite log, which is what makes the sequence
restartable rather than a script that has to complete in one process: kill it during the
diagnosis and the next process picks the diagnosis up, kill it during the fix and the next process
finishes the fix without repeating an action.

The two agents are deliberately unequal. The diagnoser has judgment and no power: read-only tools,
and its answer is validated and reviewed before a person sees it. The fixer has power and no
judgment: it writes to the warehouse, but only the actions the person approved, enforced by the
sandbox. The rule-based graph in triage_bridge/runner.py is the baseline this is measured against
(see triage_bridge/diag_bench.py), not something this replaces.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from triage.dag import PipelineDAG
from triage.knowledge import CaseLibrary
from triage.service import TriageService

from harness import AgentSpec, Envelope, Harness, OpenAICompatModel, Policy, RecoveryPolicy, RunStore

from . import diagnoser
from .diagnoser import Diagnosis, derive_plan
from .runner import DATA, FIXER, KEY_FIELDS, TOOLS, Incomplete, NeedsReview, describe_task, plan_to_calls
from .warehouse import Warehouse


class Escalated(Exception):
    """No plan to approve: the diagnosis never arrived, or the reviewer refused to put it to a person."""


class TwoAgentTriage:
    def __init__(self, workdir: str | Path, model=None, recovery: RecoveryPolicy | None = None,
                 max_retries: int = 2, fault_hook=None, max_steps: int = 14, communicator=None):
        self.dir = Path(workdir)
        (self.dir / "cases").mkdir(parents=True, exist_ok=True)
        self.dag = PipelineDAG.from_yaml(DATA / "pipeline.yaml")
        self.library = CaseLibrary.load(DATA / "case_library.yaml")
        self.warehouse = Warehouse(self.dir / "warehouse.db")
        self.store = RunStore(self.dir / "harness_runs.db")
        self.model = model or OpenAICompatModel()
        self.recovery = recovery or RecoveryPolicy()
        self.max_retries = max_retries
        self.fault_hook = fault_hook
        self.max_steps = max_steps
        self._rules: dict[str, dict] = {}
        self.communicator = communicator      # writes the briefing at the gate from the tracer's findings

    def close(self) -> None:
        """Release the run log. Needed wherever the working directory is deleted after, since
        Windows refuses to delete a file that is still open."""
        self.store.close()

    def __enter__(self) -> TwoAgentTriage:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ durable case state

    def _case_path(self, cid: str) -> Path:
        return self.dir / "cases" / f"{cid}.json"

    def _case(self, cid: str) -> dict:
        return json.loads(self._case_path(cid).read_text())

    def _save_case(self, cid: str, **fields) -> dict:
        case = self._case(cid) if self._case_path(cid).exists() else {}
        case.update(fields)
        self._case_path(cid).write_text(json.dumps(case, indent=1))
        return case

    def diagnosis(self, cid: str) -> Diagnosis | None:
        """Read the diagnosis back out of the run log, so a new process does not need the first
        one's memory and cannot invent one that was never submitted."""
        for s in reversed(self.store.steps(f"diag-{cid}")):
            if s["tool"] == "submit_diagnosis" and s["status"] == "done":
                a = s["args"]
                return Diagnosis(a["category"], a["fix_target"], a.get("cause") or "",
                                 (a.get("reason") or "").strip(), derive_plan(self.dag, a["fix_target"]))
        return None

    # ------------------------------------------------------------------ 1. the model diagnoses

    def diagnose(self, incident: dict) -> dict:
        cid = incident["id"]
        self._save_case(cid, incident=incident)
        rid = f"diag-{cid}"
        spec = diagnoser.spec(self.max_steps)
        reg = diagnoser.tools(self.dag, self.library, incident)
        h = Harness(reg, self.model, self.store, fault_hook=self.fault_hook)
        if self.store.get_run(rid) is None:
            h.start(spec, diagnoser.task(incident), task_kind="triage_diagnose", run_id=rid)
        else:
            h.resume(spec, rid)
        return self.view(cid)

    def rules(self, cid: str) -> dict:
        """What the deterministic triage graph makes of the same incident: the tracer's root cause and
        the rules' category. It changes nothing (the graph stops before its gate), and it is the
        cross-check the model's diagnosis has to agree with before a person is asked to approve it."""
        if cid not in self._rules:
            inc = dict(self._case(cid)["incident"])
            inc.pop("label", None)
            lib = CaseLibrary(copy.deepcopy(self.library.cases))       # its learn step must not touch ours
            self._rules[cid] = TriageService(self.dag, lib, communicator=self.communicator).open(inc)[1]
        return self._rules[cid]

    def review(self, cid: str) -> list[str]:
        """Checks before a person sees the model's diagnosis as something to approve: the tracer has to
        agree on the root cause, and the rules have to recognise the failure. The category is the
        rules', because a model choosing from a list puts a familiar name on a failure nobody has seen
        (pipeline-triage-graph measured that on incidents 41-60)."""
        d = self.diagnosis(cid)
        if d is None:
            return ["the diagnoser submitted nothing"]
        base = self.rules(cid)
        issues = []
        if d.fix_target != base.get("fix_target"):
            issues.append(f"diagnoser and tracer disagree on the root cause ({d.fix_target} vs "
                          f"{base.get('fix_target')}); a person should decide from both")
        if base.get("category") == "unknown":
            issues.append("unrecognised failure category")
        if not d.reason:
            issues.append("no proposal to review")
        if d.fix_target in self.dag.protected:
            issues.append(f"{d.fix_target} is protected")
        return issues

    def view(self, cid: str) -> dict:
        d, issues = self.diagnosis(cid), self.review(cid)
        decision = self._case(cid).get("decision") if self._case_path(cid).exists() else None
        apply_run = self.store.get_run(f"apply-{cid}")
        briefing = self.rules(cid).get("briefing") if d is not None and self.communicator is not None else None
        return {"case_id": cid, "diagnosis": d and vars(d), "review_issues": issues, "briefing": briefing,
                "waiting_for_human": not issues and decision is None,
                "decision": decision, "apply_status": apply_run and apply_run["status"],
                "outcome": ("escalated" if issues or decision == "reject"
                            else "resolved" if apply_run and apply_run["status"] == "done"
                            else "pending")}

    # ------------------------------------------------------------------ 2. a person decides

    def decide(self, cid: str, approve: bool, reviewer: str = "on-call") -> dict:
        issues = self.review(cid)
        if issues:
            self._save_case(cid, decision="escalate", reviewer=reviewer)
            raise Escalated("; ".join(issues))
        self._save_case(cid, decision="approve" if approve else "reject", reviewer=reviewer)
        return self.apply(cid) if approve else self.view(cid)

    # ------------------------------------------------------------------ 3. the model applies it

    def apply(self, cid: str) -> dict:
        d = self.diagnosis(cid)
        if d is None or self._case(cid).get("decision") != "approve":
            raise Escalated(f"{cid} has no approved diagnosis")
        calls = plan_to_calls(self.dag, cid, self.rules(cid)["category"], d.plan, d.reason)
        envelope = Envelope.from_calls(calls, KEY_FIELDS)
        agent = AgentSpec("fixer", FIXER, Policy(set(TOOLS), envelope=envelope),
                          max_steps=len(calls) + 5, max_retries=self.max_retries)
        rid = f"apply-{cid}"
        h = Harness(self.warehouse.tools(), self.model, self.store, fault_hook=self.fault_hook,
                    recovery=self.recovery, verifier=self.warehouse.took_effect)
        if self.store.get_run(rid) is None:
            view = {"category": d.category, "proposal": d.reason, "plan": d.plan}
            h.start(agent, describe_task(cid, view, envelope), task_kind="triage_apply", run_id=rid)
        else:
            h.resume(agent, rid)
        run = self.store.get_run(rid)
        if run["status"] == "needs_review":
            raise NeedsReview(rid, self.store.steps(rid)[-1])
        if run["status"] == "failed":
            raise Incomplete(f"{rid}: {run['answer']}")
        return self.view(cid)

    # ------------------------------------------------------------------ after a crash

    def recover(self, cid: str) -> dict:
        """Pick the case up wherever it was left, in a process that has none of the first one's state."""
        case = self._case(cid)
        diag = self.store.get_run(f"diag-{cid}")
        if diag is None or diag["status"] not in ("done", "failed"):
            self.diagnose(case["incident"])
        if case.get("decision") == "approve":
            return self.apply(cid)
        return self.view(cid)

    def resolve(self, cid: str, took_effect: bool) -> dict:
        """A person checked the interrupted step and says whether it went through."""
        rid = f"apply-{cid}"
        last = self.store.steps(rid)[-1]
        self.store.resolve_review(rid, last["idx"], took_effect)
        return self.apply(cid)

    def trace(self, cid: str) -> list[str]:
        """Both runs as one readable sequence, which is what an operator asks for after an incident."""
        out = []
        for rid, label in ((f"diag-{cid}", "diagnose"), (f"apply-{cid}", "apply")):
            if self.store.get_run(rid) is None:
                continue
            out.append(f"[{label}] {self.store.get_run(rid)['status']}")
            out += [f"  {s['idx']} {s['tool']}({json.dumps(s['args'])[:80]}) -> {s['status']}"
                    + (f" [{s['error']}]" if s.get("error") else "") for s in self.store.steps(rid)]
        return out
