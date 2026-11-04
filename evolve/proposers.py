"""Who suggests the next configuration to try.

Two of them, same interface. `RuleProposer` reads the benchmark's own record of which tool
caused the most human interruptions and aims there first; it is deterministic, so CI gets the
same run every time. `ModelProposer` hands the same evidence to a local open-weight model and
lets it pick, which is the interesting version and needs Ollama running.

The model proposer is itself an agent inside the harness, with one tool and nothing else in
its allowlist. It cannot touch the benchmark or the warehouse because the sandbox will not
let it: whatever it suggests comes back as a `propose_change` call and is then checked by
`evolve.guard` like any other candidate.
"""
from __future__ import annotations

import json
import os
from typing import Iterable, Protocol

from pydantic import BaseModel

from harness import AgentSpec, Harness, OpenAICompatModel, Policy, RunStore, ToolRegistry

from .knobs import CONFIDENCE, NON_IDEMPOTENT, RETRIES, Knobs, neighbours
from .scorer import Score

PREFERRED = ("unique_payload", "pause", "assume_done", "retry")


class Proposer(Protocol):
    name: str

    def propose(self, best: Score, tried: set[Knobs]) -> list[Knobs]:
        ...


def _fresh(cands: Iterable[Knobs], tried: set[Knobs]) -> list[Knobs]:
    out, seen = [], set(tried)
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


class RuleProposer:
    """Aim at the tool that is actually costing the interruptions, then fall back to the rest
    of the space. No model, no network, same answer every run."""

    name = "rules"

    def propose(self, best: Score, tried: set[Knobs]) -> list[Knobs]:
        ranked = sorted(best.paused_on.items(), key=lambda kv: -kv[1])
        first = [best.knobs.with_recovery(tool, s)
                 for tool, _ in ranked if tool in NON_IDEMPOTENT
                 for s in PREFERRED
                 if best.knobs.recovery_map.get(tool, "pause") != s]
        return _fresh([*first, *neighbours(best.knobs)], tried)


class Change(BaseModel):
    tool: str
    strategy: str
    reason: str


class ModelProposer:
    """A local open-weight model, given the same evidence, proposing one change at a time."""

    name = "model"

    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or os.environ.get("HARNESS_MODEL", "qwen3:8b")
        self.base_url = base_url or os.environ.get("HARNESS_BASE_URL", "http://localhost:11434/v1")
        self.fallback = RuleProposer()

    def _registry(self, picked: list[Change]) -> ToolRegistry:
        reg = ToolRegistry()

        @reg.tool("propose_change", "Propose one change to the recovery strategy for one tool.",
                  Change, idempotent=True, side_effects=False)
        def propose_change(tool: str, strategy: str, reason: str) -> str:
            picked.append(Change(tool=tool, strategy=strategy, reason=reason))
            return f"recorded: {tool} -> {strategy}"

        return reg

    def propose(self, best: Score, tried: set[Knobs]) -> list[Knobs]:
        picked: list[Change] = []
        task = json.dumps({
            "current_config": best.knobs.to_dict(),
            "result": {"finished_without_a_human": f"{best.unattended}/{best.reached_gate}",
                       "interruptions_by_tool": best.paused_on,
                       "duplicated_actions": best.duplicates, "dropped_actions": best.dropped},
            "tools_not_safe_to_repeat": list(NON_IDEMPOTENT),
            "strategies": {
                "pause": "stop and ask a person. never wrong, always costs an interruption",
                "unique_payload": "ask the warehouse whether this exact call is recorded, but only when no "
                                  "earlier step in the run made the identical call",
                "assume_done": "treat it as finished without checking",
                "retry": "run it again without checking"},
            "goal": "raise finished_without_a_human. duplicated_actions and dropped_actions must stay 0 "
                    "or the change is thrown away.",
            "already_tried": [k.to_dict() for k in list(tried)[-12:]],
        }, indent=1)
        agent = AgentSpec(
            "proposer",
            "You tune one setting of a crash-recovery policy. Call propose_change exactly once with the "
            "single change most likely to reduce human interruptions without ever duplicating or dropping "
            "an action. Then stop.",
            Policy({"propose_change"}), max_steps=3)
        try:
            Harness(self._registry(picked), OpenAICompatModel(self.model, self.base_url), RunStore()).start(
                agent, task, task_kind="evolve_proposal")
        except Exception as e:  # no Ollama, bad json, model refused: fall back rather than stall the loop
            print(f"  model proposer unavailable ({type(e).__name__}: {e}); using rules")
            return self.fallback.propose(best, tried)

        cands = []
        for c in picked:
            if c.tool in NON_IDEMPOTENT:
                cands.append(best.knobs.with_recovery(c.tool, c.strategy))
        if not cands:
            return self.fallback.propose(best, tried)
        # The model gets its pick tried first; the rule list keeps the loop moving if it was wrong.
        return _fresh([*cands, *self.fallback.propose(best, tried)], tried)


def get(name: str) -> Proposer:
    return {"rules": RuleProposer, "model": ModelProposer}[name]()
