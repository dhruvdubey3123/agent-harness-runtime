"""What the harness does with a step a crash left in flight.

A step written as 'dispatched' may or may not have taken effect. If the tool is safe to
repeat, the harness just repeats it. If it is not, the harness has to decide between three
bad options: repeat it (risk a duplicate), skip it (risk a dropped action), or stop and ask
a person. Stopping is always correct, and it costs a human interruption every time.

This module makes that decision a policy instead of a hardcoded rule, so it can be tuned and,
in `evolve`, searched. The strategies:

* `pause`          always stop for a person. The default, and never wrong.
* `unique_payload` ask the environment whether this exact call is recorded there, but only
                   when no earlier step in the same run made the identical call. Without that
                   guard, evidence left by the earlier call would be read as evidence for this
                   one and the action would be silently dropped.
* `assume_done`    treat it as finished. Safe only for a tool whose effect does not matter.
* `retry`          run it again. Safe only for a tool that is in fact repeatable.

The last two exist so the search space contains unsafe options: a policy search is only
meaningful if its safety gate can actually reject something.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

STRATEGIES = ("pause", "unique_payload", "assume_done", "retry")

# Given a tool name and its arguments, does the environment show this call took effect?
# True / False are evidence; None means the environment cannot tell.
Verifier = Callable[[str, dict], "bool | None"]


@dataclass(frozen=True)
class RecoveryPolicy:
    per_tool: Mapping[str, str] = field(default_factory=dict)
    default: str = "pause"

    def __post_init__(self) -> None:
        bad = {s for s in (*self.per_tool.values(), self.default) if s not in STRATEGIES}
        if bad:
            raise ValueError(f"unknown recovery strategy: {sorted(bad)}")

    def strategy_for(self, tool: str) -> str:
        return self.per_tool.get(tool, self.default)


def decide(policy: RecoveryPolicy, verifier: Verifier | None, step: dict, steps: list[dict]) -> str:
    """-> 'pause' | 'verified_done' | 'repeat'. Called only for a non-idempotent step left dispatched."""
    strategy = policy.strategy_for(step["tool"])
    if strategy == "retry":
        return "repeat"
    if strategy == "assume_done":
        return "verified_done"
    if strategy != "unique_payload" or verifier is None:
        return "pause"
    # An identical call earlier in this run would have left the same trace, so the evidence
    # would not be about this step. Refuse to guess.
    earlier = [s for s in steps if s["idx"] != step["idx"] and s["tool"] == step["tool"] and s["args"] == step["args"]]
    if earlier:
        return "pause"
    evidence = verifier(step["tool"], step["args"])
    if evidence is True:
        return "verified_done"
    if evidence is False:
        return "repeat"
    return "pause"
