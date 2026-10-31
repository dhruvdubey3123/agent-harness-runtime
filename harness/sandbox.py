"""Least-privilege execution. Every tool call goes through here.

A policy is per agent: which tools it may call, how often, how long each call may take,
and which argument values are off limits. The model never touches a tool directly.
"""
from __future__ import annotations

import concurrent.futures as cf
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from .tools import ToolRegistry
from .types import RetryableToolError, ToolCall, ToolDenied


@dataclass
class Policy:
    allowed_tools: set[str]
    max_calls_per_tool: int = 50
    timeout_s: float = 30.0
    denied_path_parts: set[str] = field(default_factory=lambda: {"..", ".git", ".env", ".ssh"})


class Sandbox:
    def __init__(self, registry: ToolRegistry, policy: Policy):
        self.registry = registry
        self.policy = policy
        self.calls: dict[str, int] = {}
        self._pool = cf.ThreadPoolExecutor(max_workers=4)

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
            return tool, tool.args_model(**call.args)
        except ValidationError as e:
            raise ToolDenied(f"invalid arguments for '{call.tool}': {e.errors()[0]['msg']}") from None

    def execute(self, call: ToolCall) -> Any:
        tool, args = self.check(call)
        self.calls[call.tool] = self.calls.get(call.tool, 0) + 1
        fut = self._pool.submit(tool.fn, **args.model_dump())
        try:
            return fut.result(timeout=self.policy.timeout_s)
        except cf.TimeoutError:
            raise RetryableToolError(f"'{call.tool}' timed out after {self.policy.timeout_s}s") from None
        except PermissionError as e:
            raise ToolDenied(str(e)) from None
