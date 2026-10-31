"""Models the harness can drive. The harness only needs one method: given the context, what next?"""
from __future__ import annotations

import os
from typing import Protocol

from .types import Action, Finish, ToolCall


class Model(Protocol):
    def next_action(self, context: list[dict], tools: list[dict]) -> Action: ...


class ScriptedModel:
    """Deterministic stand-in for an LLM, used by tests and benchmarks.

    It picks the next action from the number of tool results already in the context, so,
    like a real model, the same context always produces the same next step. That property is
    what makes resume-after-crash meaningful to test.
    """

    def __init__(self, script: list[Action]):
        self.script = script
        self.calls = 0

    def next_action(self, context: list[dict], tools: list[dict]) -> Action:
        self.calls += 1
        n = sum(1 for m in context if m["role"] == "tool")
        return self.script[n] if n < len(self.script) else Finish("script complete")


class AnthropicModel:
    """Claude via the Messages API with tool use. Requires `pip install anthropic` and ANTHROPIC_API_KEY."""

    def __init__(self, model: str | None = None, max_tokens: int = 1024):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("HARNESS_MODEL", "claude-sonnet-4-5")
        self.max_tokens = max_tokens

    def next_action(self, context: list[dict], tools: list[dict]) -> Action:
        system = "\n\n".join(m["content"] for m in context if m["role"] == "system")
        messages: list[dict] = []
        i = 0
        for m in context:
            if m["role"] == "user":
                messages.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                i += 1
                messages.append({"role": "assistant", "content": [
                    {"type": "tool_use", "id": f"call_{i}", "name": m["tool"], "input": m["args"]}]})
            elif m["role"] == "tool":
                messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": f"call_{i}", "content": str(m["content"]),
                     "is_error": bool(m.get("error"))}]})
        resp = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, system=system,
                                           tools=tools, messages=messages)
        for block in resp.content:
            if block.type == "tool_use":
                return ToolCall(block.name, dict(block.input))
        text = "".join(getattr(b, "text", "") for b in resp.content).strip()
        return Finish(text or "(no answer)")
