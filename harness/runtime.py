"""The harness: it owns the loop, the context, the sandbox and the run log.

    model  ──decides──►  harness  ──executes──►  sandbox ──► tool
      ▲                     │
      └──── context ◄── run log (SQLite) ◄── every step, before and after

Remove the model from that picture and everything left is this file.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .context import ContextBudget, build_context, est_tokens
from .model import Model
from .recovery import RecoveryPolicy, Verifier, decide as decide_recovery
from .sandbox import Policy, Sandbox
from .skills import load_skills_index
from .store import RunStore
from .tools import ToolRegistry
from .types import DENIED, FINISH_STEP, GAVE_UP, MAX_FINISH_REJECTIONS, Finish, RetryableToolError, ToolCall, ToolDenied

FaultHook = Callable[[str, int, ToolCall], None]


@dataclass
class AgentSpec:
    name: str
    instructions: str
    policy: Policy
    max_steps: int = 25
    max_retries: int = 2
    backoff_s: float = 0.05
    finish_on: frozenset[str] = frozenset()
    """Tools whose first successful call ends the run. A model told to "call this once and stop"
    will sometimes keep going and change its mind, and the harness is what enforces "once"."""


@dataclass
class RunMetrics:
    steps: int = 0
    tool_calls: int = 0
    replayed_from_log: int = 0
    retries: int = 0
    denials: int = 0
    tool_errors: int = 0
    compactions: int = 0
    auto_verified: int = 0
    est_context_tokens: list[int] = field(default_factory=list)
    wall_ms: float = 0.0


class Harness:
    def __init__(self, registry: ToolRegistry, model: Model, store: RunStore | None = None,
                 budget: ContextBudget | None = None, skills_dir: str | None = None,
                 fault_hook: FaultHook | None = None, sleep: Callable[[float], None] = time.sleep,
                 recovery: RecoveryPolicy | None = None, verifier: Verifier | None = None):
        self.registry = registry
        self.model = model
        self.store = store or RunStore()
        self.budget = budget or ContextBudget()
        self.skills_dir = skills_dir
        self.fault_hook = fault_hook
        self.recovery = recovery or RecoveryPolicy()
        self.verifier = verifier
        self.sleep = sleep
        self.metrics: dict[str, RunMetrics] = {}

    # ------------------------------------------------------------------ public API

    def start(self, agent: AgentSpec, task: str, task_kind: str | None = None, run_id: str | None = None) -> str:
        rid = self.store.create_run(agent.name, task, task_kind, run_id)
        self._loop(agent, rid)
        return rid

    def resume(self, agent: AgentSpec, run_id: str) -> str:
        run = self.store.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        if run["status"] in ("done", "failed"):
            return run_id
        self.store.set_status(run_id, "running")
        self._loop(agent, run_id, resuming=True)
        return run_id

    # ------------------------------------------------------------------ loop

    @staticmethod
    def _ended_by(agent: AgentSpec, steps: list[dict]) -> str | None:
        for s in steps:
            if s["tool"] in agent.finish_on and s["status"] == "done":
                return str(s["result"])
        return None

    def _hook(self, event: str, idx: int, call: ToolCall) -> None:
        if self.fault_hook:
            self.fault_hook(event, idx, call)

    def _loop(self, agent: AgentSpec, run_id: str, resuming: bool = False) -> None:
        with Sandbox(self.registry, agent.policy) as sandbox:
            self._run_loop(agent, run_id, sandbox, resuming)

    def _run_loop(self, agent: AgentSpec, run_id: str, sandbox: Sandbox, resuming: bool) -> None:
        m = self.metrics.setdefault(run_id, RunMetrics())
        t0 = time.perf_counter()
        run = self.store.get_run(run_id)
        steps = self.store.steps(run_id)
        sandbox.restore(steps)  # per-tool call budgets, and the approved plan, as the log left them
        if resuming:
            m.replayed_from_log += sum(1 for s in steps if s["status"] in ("done", "error"))

        # Recovery: the crash may have hit while a tool was running or before its result was saved.
        if steps and steps[-1]["status"] in ("started", "dispatched"):
            last = steps[-1]
            action = "repeat"
            if last["status"] == "dispatched" and not last["idempotent"]:
                action = decide_recovery(self.recovery, self.verifier, last, steps)
            if action == "pause":
                self.store.set_status(run_id, "needs_review",
                                      f"step {last['idx']} ({last['tool']}) was interrupted and is not safe to repeat; "
                                      "check whether it took effect, then mark it done or retry it")
                m.wall_ms += (time.perf_counter() - t0) * 1000
                return
            if action == "verified_done":
                m.auto_verified += 1
                self.store.resolve_review(run_id, last["idx"], True, "verified from the environment, not by a person")
                sandbox.account({**last, "status": "done", "error": None})   # settled, so the plan is spent
            else:
                self._execute(agent, sandbox, run_id, last["idx"], ToolCall(last["tool"], last["args"]), m)
            steps = self.store.steps(run_id)

        ended = self._ended_by(agent, steps)
        if ended is not None:   # a crash landed between the tool finishing and the status being written
            self.store.set_status(run_id, "done", ended)
            m.steps = len(steps)
            m.wall_ms += (time.perf_counter() - t0) * 1000
            return

        skills = load_skills_index(self.skills_dir, run["task_kind"]) if self.skills_dir else ""
        idx = len(steps)
        while idx < agent.max_steps:
            ctx, compacted = build_context(agent.instructions, run["task"], steps, self.budget, skills)
            m.compactions += compacted
            m.est_context_tokens.append(est_tokens(ctx))
            try:
                action = self.model.next_action(ctx, sandbox.visible_tools())
            except Exception as e:
                # The model timed out or is unreachable. Nothing was decided and nothing is guessed: the run
                # stops with everything so far in the log and resumes from there when the model is back,
                # or a person picks it up. A crash of the process (BaseException) still propagates.
                self.store.set_status(run_id, "model_unavailable", f"model unavailable ({type(e).__name__}): {e}"[:400])
                break
            if isinstance(action, Finish):
                left = sandbox.unspent()
                if left and sum(1 for s in steps if s["tool"] == FINISH_STEP) < MAX_FINISH_REJECTIONS:
                    # The envelope is a ceiling, so a model can stop early and the run still looks
                    # complete. Approved work that never happened is a dropped action, so a
                    # finish with approved actions outstanding goes back to the model, in the log.
                    self.store.begin_step(run_id, idx, FINISH_STEP, {"answer": action.answer}, True)
                    self.store.finish_step(run_id, idx, error=f"{DENIED}not finished: approved actions "
                                                              f"still to apply: {'; '.join(left)}")
                    steps = self.store.steps(run_id)
                    idx += 1
                    continue
                if left:
                    self.store.set_status(run_id, "failed", "finished with approved actions not applied: "
                                          + "; ".join(left))
                else:
                    self.store.set_status(run_id, "done", action.answer)
                break
            self._execute(agent, sandbox, run_id, idx, action, m)
            steps = self.store.steps(run_id)
            ended = self._ended_by(agent, steps)
            if ended is not None:
                self.store.set_status(run_id, "done", ended)
                break
            idx += 1
        else:
            self.store.set_status(run_id, "failed", f"hit max_steps={agent.max_steps}")
        m.steps = len(self.store.steps(run_id))
        m.wall_ms += (time.perf_counter() - t0) * 1000

    def _execute(self, agent: AgentSpec, sandbox: Sandbox, run_id: str, idx: int, call: ToolCall, m: RunMetrics) -> None:
        call = sandbox.bind(call)   # the log records the approved values, so resume and verification see them
        tool = self.registry.get(call.tool)
        self.store.begin_step(run_id, idx, call.tool, call.args, tool.idempotent if tool else True)
        self._hook("before_tool", idx, call)
        for attempt in range(agent.max_retries + 1):
            self.store.bump_attempts(run_id, idx)
            try:
                m.tool_calls += 1
                sandbox.check(call)  # a denied call never reaches 'dispatched'
                self.store.mark_dispatched(run_id, idx)
                self._hook("during_tool", idx, call)
                result = sandbox.execute(call)
            except ToolDenied as e:
                m.denials += 1
                self.store.finish_step(run_id, idx, error=f"{DENIED}{e}")
                return
            except RetryableToolError as e:
                if attempt < agent.max_retries:
                    m.retries += 1
                    self.sleep(agent.backoff_s * (2 ** attempt))
                    continue
                m.tool_errors += 1
                self.store.finish_step(run_id, idx, error=f"{GAVE_UP}{attempt + 1} attempts: {e}")
                return
            except Exception as e:  # tool bug: report to the model instead of killing the run
                m.tool_errors += 1
                self.store.finish_step(run_id, idx, error=f"{type(e).__name__}: {e}")
                return
            # The dangerous window: the tool has run, but the log does not know yet.
            self._hook("after_tool", idx, call)
            self.store.finish_step(run_id, idx, result=result)
            return
