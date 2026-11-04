"""The approved plan as a boundary instead of a script.

A person approves a plan. Replaying that plan call by call is safe but it is not an agent: nothing
decides anything. Letting a model carry the plan out is an agent, but then the thing a person
approved and the thing that happens can drift apart.

The envelope is the line between the two. The model chooses the order, the wording, what to do
about an error and when to stop. Every call it makes has to match an action a person approved and
still have budget left. Anything else is refused by the sandbox before it reaches the tool and is
written to the run log, so "the model stayed inside the approved plan" is a property of the
runtime rather than a hope about the prompt.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .types import ToolCall, ToolDenied


@dataclass(frozen=True)
class ApprovedAction:
    """One action a person approved, and how many times it may happen.

    `match` holds the fields that decide what the action does to the world: which node is rerun,
    which case is commented on. Those have to match exactly. Everything else is the model's to
    write, which is the point of using a model at all: the wording of a resolution comment is its
    job, the choice of what to touch is not.
    """

    tool: str
    match: tuple[tuple[str, Any], ...] = ()
    times: int = 1

    def covers(self, call: ToolCall) -> bool:
        return call.tool == self.tool and all(call.args.get(k) == v for k, v in self.match)

    def __str__(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in self.match)
        return f"{self.tool}({inner})" + (f" x{self.times}" if self.times > 1 else "")


@dataclass(frozen=True)
class Envelope:
    actions: tuple[ApprovedAction, ...]

    @classmethod
    def from_calls(cls, calls: Sequence[ToolCall], key_fields: Mapping[str, Sequence[str]]) -> Envelope:
        """Build an envelope from the concrete calls a plan expands to.

        Identical actions collapse into one entry with a count, so a plan that legitimately reruns
        the same job twice approves it twice and a model that does it twice on its own is refused.
        """
        counted: dict[tuple[str, tuple[tuple[str, Any], ...]], int] = {}
        for c in calls:
            keys = tuple((k, c.args[k]) for k in key_fields.get(c.tool, ()) if k in c.args)
            counted[(c.tool, keys)] = counted.get((c.tool, keys), 0) + 1
        return cls(tuple(ApprovedAction(tool, keys, n) for (tool, keys), n in counted.items()))

    def check(self, call: ToolCall, used: Mapping[int, int]) -> int:
        """Index of an approved action this call may be charged to, or ToolDenied with the reason.

        Pure: the caller records the charge, so the same check can run before dispatch, again at
        dispatch, and once more over the log when a crashed run is resumed.
        """
        spent = None
        for i, a in enumerate(self.actions):
            if a.covers(call):
                if used.get(i, 0) < a.times:
                    return i
                spent = a
        if spent is not None:
            raise ToolDenied(f"the approved plan allows {spent} and that is already applied; "
                             "a person has to approve another one")
        raise ToolDenied(f"'{call.tool}' with those arguments is not in the approved plan. "
                         f"Approved: {self.describe()}")

    def remaining(self, used: Mapping[int, int]) -> list[str]:
        """Approved actions not yet applied, as text. The envelope is a ceiling, and this is the floor."""
        out = []
        for i, a in enumerate(self.actions):
            left = a.times - used.get(i, 0)
            if left > 0:
                out.append(str(ApprovedAction(a.tool, a.match, left)))
        return out

    def describe(self) -> str:
        return "; ".join(str(a) for a in self.actions) or "(nothing)"
