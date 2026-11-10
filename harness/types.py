from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ToolCall:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Finish:
    answer: str


Action = ToolCall | Finish

RunStatus = Literal["running", "done", "failed", "needs_review", "model_unavailable"]
StepStatus = Literal["started", "dispatched", "done", "error"]

# How a failed step says what it did to the world. Both of these mean "nothing happened", which
# is what lets a resumed run tell them apart from a step whose effect is unknown.
DENIED = "denied: "
GAVE_UP = "gave up after "

# A finish the harness sent back because approved work was still outstanding.
FINISH_STEP = "finish"
MAX_FINISH_REJECTIONS = 2


class SimulatedCrash(BaseException):
    """Raised in tests and benchmarks to kill a run mid-step, like a process dying.

    It derives from BaseException so ordinary `except Exception` handlers (including the
    retry loop) cannot swallow it, which is how a real crash behaves.
    """


class ToolDenied(Exception):
    """The sandbox refused a call. Never retried."""


class RetryableToolError(Exception):
    """A transient failure the tool reports itself (rate limit, flaky network), or a timeout on a tool
    that is safe to repeat. Retried with backoff."""


class EffectUnknown(Exception):
    """A call that is not safe to repeat ran out of time. Its thread is still running and may yet take
    effect, so it is neither retried nor reported as having done nothing: the run stops for a person,
    exactly as it does when a crash leaves such a step in flight."""
