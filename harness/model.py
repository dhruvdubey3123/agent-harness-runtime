"""Models the harness can drive. The harness only needs one method: given the context, what next?"""
from __future__ import annotations

import json
import os
import re
import sys
import time
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


def _rate_limited(r) -> bool:
    """A hosted tier saying "slow down" (429) or "busy" (503). Not the model failing."""
    return r.status_code in (429, 503)


def _unparsable(r) -> bool:
    """The provider could not turn the model's output into a tool call, and answered 400 instead of a
    message. Seen with gpt-oss-120b on Groq. Nothing was executed, so the same question can be asked
    again, and a sampled model usually gets it right the next time."""
    return r.status_code == 400 and "could not be parsed" in (getattr(r, "text", "") or "")


def _retry_after(r) -> float | None:
    """How long the provider asked us to wait, in seconds. Groq sends it as a number of seconds, and
    says the same thing in prose ("Please try again in 7m2.5s") when the header is missing."""
    header = r.headers.get("retry-after") if hasattr(r, "headers") else None
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", getattr(r, "text", "") or "")
    return int(m.group(1) or 0) * 60 + float(m.group(2)) if m else None


def _too_big_to_ever_fit(r) -> bool:
    """One request that needs more tokens than the whole per-minute budget. No amount of waiting makes
    it fit, so retrying is a slow way of failing. Groq spells this out: "Limit 8000, ... Requested 9000"."""
    m = re.search(r"Limit (\d+),.*?Requested (\d+)", (getattr(r, "text", "") or ""), re.S)
    return bool(m) and int(m.group(2)) > int(m.group(1))


def _wait(r, attempt: int, max_wait: float) -> float | None:
    """Seconds to sleep before asking again, or None to stop trying.

    A provider that asks for longer than we are willing to wait, or that cannot fit this request into
    its budget at all, is telling us to change something rather than to be patient.
    """
    if not _rate_limited(r):
        return 0.5
    if _too_big_to_ever_fit(r):
        return None
    asked = _retry_after(r)
    if asked is None:
        return min(2 ** attempt * 2, max_wait)
    return asked if asked <= max_wait else None

def _note(r, pause: float) -> None:
    """A long wait with no output looks like a hang, so say what is happening."""
    if pause >= 5:
        why = "rate limited" if _rate_limited(r) else "output could not be parsed"
        print(f"  [{why}, waiting {pause:.0f}s]", file=sys.stderr, flush=True)


def _raise_with_body(r) -> None:
    """Raise with what the provider actually said. A hosted API answering 400 names the part of the
    request it rejected, and a bare status line turns that into guesswork."""
    import httpx

    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        body = (getattr(r, "text", "") or "")[:600]
        try:
            request = e.request
        except RuntimeError:            # never sent, so there is no request to point at
            request = None
        raise httpx.HTTPStatusError(f"{e}\n{body}" if body else str(e), request=request, response=e.response) from None


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


class OpenAICompatModel:
    """Any server that speaks the OpenAI chat-completions API with tool calling: Ollama, llama.cpp's
    llama-server, vLLM, LM Studio. Defaults to a local Ollama, so open-weight models run with no API key.

        ollama pull qwen3:8b
        HARNESS_MODEL=qwen3:8b python examples/real_model.py
    """

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 timeout: float | None = None, temperature: float = 0.0):
        self.model = model or os.environ.get("HARNESS_MODEL", "qwen3:8b")
        self.base_url = (base_url or os.environ.get("HARNESS_BASE_URL", "http://localhost:11434/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("HARNESS_API_KEY", "not-needed-for-local")
        # Per request. A small local model thinking at length on a laptop can take minutes, so it is
        # configurable; when it is exceeded the step treats the model as unavailable and the run pauses.
        self.timeout = timeout or float(os.environ.get("HARNESS_TIMEOUT", "300"))
        self.temperature = temperature
        self.retries = int(os.environ.get("HARNESS_RETRIES", "6"))
        self.max_wait = float(os.environ.get("HARNESS_MAX_WAIT", "180"))

    @staticmethod
    def _tools(tools: list[dict]) -> list[dict]:
        return [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                  "parameters": t.get("input_schema", {"type": "object"})}}
                for t in tools]

    @staticmethod
    def _messages(context: list[dict]) -> list[dict]:
        out: list[dict] = []
        i = 0
        for m in context:
            if m["role"] in ("system", "user"):
                out.append({"role": m["role"], "content": str(m["content"])})
            elif m["role"] == "assistant":
                i += 1
                out.append({"role": "assistant", "content": "", "tool_calls": [{
                    "id": f"call_{i}", "type": "function",
                    "function": {"name": m["tool"], "arguments": json.dumps(m["args"])}}]})
            elif m["role"] == "tool":
                content = str(m["content"])
                out.append({"role": "tool", "tool_call_id": f"call_{i}",
                            "content": ("ERROR: " + content) if m.get("error") else content})
        return out

    def next_action(self, context: list[dict], tools: list[dict]) -> Action:
        import httpx

        body = {"model": self.model, "messages": self._messages(context), "tools": self._tools(tools),
                "temperature": self.temperature, "stream": False}
        for attempt in range(self.retries + 1):
            r = httpx.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout,
                           headers={"Authorization": f"Bearer {self.api_key}"})
            if attempt < self.retries and (_rate_limited(r) or _unparsable(r)):
                pause = _wait(r, attempt, self.max_wait)
                if pause is not None:
                    _note(r, pause)
                    time.sleep(pause)
                    continue
            break
        _raise_with_body(r)
        msg = r.json()["choices"][0]["message"]
        for call in msg.get("tool_calls") or []:
            fn = call["function"]
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    args = {"_unparsed": args}   # the sandbox's validation turns this into an error the model sees
            return ToolCall(fn["name"], args)
        # reasoning models (Qwen3, DeepSeek-R1) may wrap their thinking in <think> tags; keep only the answer
        text = re.sub(r"<think>.*?</think>", "", msg.get("content") or "", flags=re.S).strip()
        return Finish(text or "(no answer)")
