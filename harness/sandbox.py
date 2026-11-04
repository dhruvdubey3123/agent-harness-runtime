"""Least-privilege execution. Every tool call goes through here.

A policy is per agent: which tools it may call, how often, how long each call may take,
and which argument values are off limits. The model never touches a tool directly.
"""
from __future__ import annotations

import concurrent.futures as cf
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from .envelope import Envelope
from .tools import ToolRegistry
from .types import DENIED, GAVE_UP, RetryableToolError, ToolCall, ToolDenied


@dataclass
class Policy:
    allowed_tools: set[str]
    max_calls_per_tool: int = 50
    timeout_s: float = 30.0
    denied_path_parts: set[str] = field(default_factory=lambda: {"..", ".git", ".env", ".ssh"})
    envelope: Envelope | None = None
    """The actions a person approved. Set it and the agent may still decide how to work, but it
    cannot act on anything outside the approval. Left unset, the policy is the tool list alone."""


class Sandbox:
    def __init__(self, registry: ToolRegistry, policy: Policy):
        self.registry = registry
        self.policy = policy
        self.calls: dict[str, int] = {}
        self.used: dict[int, int] = {}   # approved action index -> times already charged
        self._pool = cf.ThreadPoolExecutor(max_workers=4)

    def unspent(self) -> list[str]:
        """Approved actions that have not happened yet. Empty when there is no envelope."""
        return self.policy.envelope.remaining(self.used) if self.policy.envelope is not None else []

    def close(self) -> None:
        """Let the worker threads go. One run means one sandbox, and a benchmark means hundreds of
        runs, so a pool that outlives its run is a leak that only shows up at scale."""
        self._pool.shutdown(wait=False)

    def __enter__(self) -> Sandbox:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def restore(self, steps: list[dict]) -> None:
        """Rebuild the budgets from the run log, so a resumed run cannot spend them twice.

        A call that was refused never happened, and a call that gave up after retrying said it had
        no effect, so neither is charged against the approved plan. Anything else is, including a
        step that failed in a way the tool did not explain: if the effect is unknown, the
        conservative reading is that it happened.
        """
        for s in steps:
            if s["status"] not in ("done", "error"):
                continue
            self.calls[s["tool"]] = self.calls.get(s["tool"], 0) + 1
            err = str(s.get("error") or "")
            if self.policy.envelope is None or err.startswith(DENIED) or err.startswith(GAVE_UP):
                continue
            try:
                self._charge(self.policy.envelope.check(ToolCall(s["tool"], s["args"]), self.used))
            except ToolDenied:
                pass   # the log already holds more of this action than the plan approved

    def _charge(self, index: int) -> None:
        self.used[index] = self.used.get(index, 0) + 1

    def visible_tools(self) -> list[dict]:
        """Only allowed tools are ever shown to the model."""
        return [self.registry.get(n).schema() for n in sorted(self.policy.allowed_tools) if self.registry.get(n)]

    def check(self, call: ToolCall) -> Any:
        if call.tool not in self.policy.allowed_tools:
            raise ToolDenied(f"tool '{call.tool}' is not allowed for this agent")
        tool = self.registry.get(call.tool)
        if tool is None:
            raise ToolDenied(f"unknown tool '{call.tool}'")
        if self.calls.get(call.tool, 0) >= self.policy.max_calls_per_tool:
            raise ToolDenied(f"call budget exhausted for '{call.tool}'")
        for v in call.args.values():
            if isinstance(v, str) and any(part in v.replace("\\", "/").split("/") for part in self.policy.denied_path_parts):
                raise ToolDenied(f"argument touches a denied path: {v}")
        try:
            args = tool.args_model(**call.args)
        except ValidationError as e:
            raise ToolDenied(f"invalid arguments for '{call.tool}': {e.errors()[0]['msg']}") from None
        if self.policy.envelope is not None:
            # Checked against the validated arguments, so a model cannot slip past the approval on
            # a type the tool would have coerced anyway ("1" and 1 are the same node).
            self.policy.envelope.check(ToolCall(call.tool, args.model_dump()), self.used)
        return tool, args

    def execute(self, call: ToolCall) -> Any:
        tool, args = self.check(call)
        self.calls[call.tool] = self.calls.get(call.tool, 0) + 1
        index = (self.policy.envelope.check(ToolCall(call.tool, args.model_dump()), self.used)
                 if self.policy.envelope is not None else None)
        fut = self._pool.submit(tool.fn, **args.model_dump())
        try:
            result = fut.result(timeout=self.policy.timeout_s)
        except cf.TimeoutError:
            # Retryable by contract: the tool is reporting that it did not get to act, so the
            # approved action is not spent and the retry is allowed to use it.
            raise RetryableToolError(f"'{call.tool}' timed out after {self.policy.timeout_s}s") from None
        except PermissionError as e:
            raise ToolDenied(str(e)) from None
        except BaseException:
            if index is not None:
                self._charge(index)   # unknown effect, so assume it happened
            raise
        if index is not None:
            self._charge(index)
        return result
