"""What stops a model from doing more than a person approved.

The envelope on its own, with no triage and no model. A fixer that misbehaves against the real
warehouse is in test_triage_bridge.py, since a real model will sometimes decide an extra job run
is a good idea and the point is that deciding it is not enough.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from harness import ApprovedAction, Envelope, Policy, Sandbox, ToolCall, ToolDenied, ToolRegistry
from harness.types import DENIED, GAVE_UP

KEYS = {"rerun_job": ("node",), "comment_ticket": ("case_id",), "append_audit": ()}
PLAN = [ToolCall("rerun_job", {"node": "sp_fill_rate"}),
        ToolCall("comment_ticket", {"case_id": "SYN-007", "body": "Resolved: patched the procedure"}),
        ToolCall("append_audit", {"line": "SYN-007: applied schema_drift fix"})]
ENV = Envelope.from_calls(PLAN, KEYS)


def test_identical_actions_collapse_into_a_count():
    twice = Envelope.from_calls([*PLAN, ToolCall("rerun_job", {"node": "sp_fill_rate"})], KEYS)
    assert twice.actions[0] == ApprovedAction("rerun_job", (("node", "sp_fill_rate"),), 2)
    assert "x2" in twice.describe()


def test_the_model_writes_the_words_and_a_person_approves_the_target():
    """A different comment body is fine. A different case is not."""
    ENV.check(ToolCall("comment_ticket", {"case_id": "SYN-007", "body": "anything I like"}), {})
    with pytest.raises(ToolDenied, match="not in the approved plan"):
        ENV.check(ToolCall("comment_ticket", {"case_id": "SYN-008", "body": "Resolved: patched the procedure"}), {})


def test_an_unapproved_action_is_refused_even_on_an_allowed_tool():
    with pytest.raises(ToolDenied, match="not in the approved plan"):
        ENV.check(ToolCall("rerun_job", {"node": "load_cases"}), {})


def test_doing_an_approved_action_twice_is_refused_separately():
    """The reason matters: it tells the operator the plan was right and the fixer overreached."""
    with pytest.raises(ToolDenied, match="already applied"):
        ENV.check(ToolCall("rerun_job", {"node": "sp_fill_rate"}), {0: 1})


# ------------------------------------------------------------------ through the sandbox

class Node(BaseModel):
    node: str


def registry(log: list[str]) -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool("rerun_job", "Run a job once.", Node, idempotent=False, side_effects=True)
    def rerun_job(node: str) -> str:
        log.append(node)
        return f"ran {node}"

    return reg


def sandbox(log: list[str], envelope: Envelope | None = ENV) -> Sandbox:
    return Sandbox(registry(log), Policy({"rerun_job"}, envelope=envelope))


def test_a_refused_call_never_reaches_the_tool():
    log: list[str] = []
    s = sandbox(log)
    with pytest.raises(ToolDenied):
        s.execute(ToolCall("rerun_job", {"node": "load_cases"}))
    assert log == []


def test_the_budget_is_spent_by_the_call_that_ran_and_not_by_the_one_refused():
    log: list[str] = []
    s = sandbox(log)
    s.execute(ToolCall("rerun_job", {"node": "sp_fill_rate"}))
    with pytest.raises(ToolDenied):
        s.execute(ToolCall("rerun_job", {"node": "load_cases"}))
    with pytest.raises(ToolDenied, match="already applied"):
        s.execute(ToolCall("rerun_job", {"node": "sp_fill_rate"}))
    assert log == ["sp_fill_rate"]


def test_no_envelope_means_the_tool_list_is_the_only_limit():
    log: list[str] = []
    s = sandbox(log, envelope=None)
    s.execute(ToolCall("rerun_job", {"node": "anything"}))
    assert log == ["anything"]


@pytest.mark.parametrize("step,spent", [
    ({"status": "done", "error": None}, True),
    ({"status": "error", "error": "ValueError: the warehouse connection dropped"}, True),
    ({"status": "error", "error": DENIED + "not in the approved plan"}, False),
    ({"status": "error", "error": GAVE_UP + "3 attempts: timed out"}, False),
])
def test_resume_charges_the_plan_the_way_the_log_describes_it(step, spent):
    """A resumed run has to agree with the log about what is left of the approval. A refusal and a
    give-up both mean nothing happened; an unexplained failure is treated as if it did."""
    log: list[str] = []
    s = sandbox(log)
    s.restore([{"tool": "rerun_job", "args": {"node": "sp_fill_rate"}, **step}])
    if spent:
        with pytest.raises(ToolDenied, match="already applied"):
            s.execute(ToolCall("rerun_job", {"node": "sp_fill_rate"}))
        assert log == []
    else:
        s.execute(ToolCall("rerun_job", {"node": "sp_fill_rate"}))
        assert log == ["sp_fill_rate"]
