"""Context assembly with a token budget owned by the harness, not the model.

Old tool outputs are the first thing to go: they are compacted to a one-line stub while the
system prompt, the task and the most recent steps stay intact. The estimate is deliberately
crude (characters / 4); swap in a real tokenizer if you need precision.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


def est_tokens(messages: list[dict]) -> int:
    return sum(len(json.dumps(m, default=str)) for m in messages) // 4


@dataclass
class ContextBudget:
    max_tokens: int = 8000
    keep_recent_steps: int = 4
    stub_chars: int = 80


def build_context(instructions: str, task: str, steps: list[dict], budget: ContextBudget,
                  skills_index: str = "") -> tuple[list[dict], int]:
    """Return (messages, number_of_compacted_results)."""
    system = instructions + (f"\n\nProcedures that worked before:\n{skills_index}" if skills_index else "")
    msgs: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    body: list[dict] = []
    for s in steps:
        if s["status"] in ("started", "dispatched"):
            continue  # interrupted step: its outcome is unknown, so the model must not see a result
        body.append({"role": "assistant", "tool": s["tool"], "args": s["args"]})
        content = s["result"] if s["status"] == "done" else f"ERROR: {s['error']}"
        body.append({"role": "tool", "content": content, "error": s["status"] == "error"})

    compacted = 0
    tool_idx = [i for i, m in enumerate(body) if m["role"] == "tool"]
    protected = set(tool_idx[-budget.keep_recent_steps:]) if budget.keep_recent_steps else set()
    for i in tool_idx:
        if est_tokens(msgs + body) <= budget.max_tokens:
            break
        if i in protected:
            continue
        text = str(body[i]["content"])
        if len(text) > budget.stub_chars:
            body[i] = {**body[i], "content": f"[compacted] {text[:budget.stub_chars]}...", "compacted": True}
            compacted += 1
    return msgs + body, compacted
