"""Whether the incident was resolved, which is a different question from whether the plan ran.

Every other test here is about the runtime: the approved actions happened, once each, nothing
extra. All of those pass for a plan aimed at the wrong node, because the runtime's job is to apply
what a person approved rather than to be right about it. These tests are about the plan.
"""
import json

import pytest

pytest.importorskip("triage")

from triage.dag import PipelineDAG  # noqa: E402

from triage_bridge import outcome  # noqa: E402
from triage_bridge.runner import DATA  # noqa: E402
from triage_bridge.warehouse import Warehouse  # noqa: E402

INC = next(json.loads(l) for l in (DATA / "incidents.jsonl").read_text().splitlines() if '"SYN-007"' in l)
DAG = PipelineDAG.from_yaml(DATA / "pipeline.yaml")

# SYN-007: load_cases dropped a column, sp_fill_rate errors on it, and the report built from
# sp_fill_rate is what the on-call engineer complained about.
PATCH, RERUN, REFRESH = "patch_procedure", "rerun_job", "refresh_table"
CORRECT = [(PATCH, "sp_fill_rate"), (RERUN, "sp_fill_rate"), (REFRESH, "fct_workforce"), (REFRESH, "rpt_workforce")]


def ran(tmp_path, plan, name="w") -> Warehouse:
    """Do a plan against a fresh warehouse, so the outcome is replayed from real recorded actions
    rather than from a list this test made up."""
    w = Warehouse(tmp_path / f"{name}.db")
    reg = w.tools()
    for tool, node in plan:
        reg.get(tool).fn(node=node, **({"change": "handle N/A in VENDOR_CODE"} if tool == PATCH else {}))
    return w


def test_the_correct_plan_resolves_the_symptom(tmp_path):
    w = ran(tmp_path, CORRECT)
    assert outcome.resolved(DAG, INC, w)
    assert outcome.unresolved(DAG, INC, w) == []
    assert outcome.explain(DAG, INC, w) == "rpt_workforce is fresh again"


def test_the_right_actions_on_the_wrong_node_do_not(tmp_path):
    """The failure mode nothing else in this repo can see. This plan is well formed, it is applied
    exactly once per action, and the procedure that is actually broken is never touched."""
    w = ran(tmp_path, [(PATCH, "sp_merge_cases"), (RERUN, "sp_merge_cases"),
                       (REFRESH, "fct_workforce"), (REFRESH, "rpt_workforce")])
    assert not outcome.resolved(DAG, INC, w)
    assert "sp_fill_rate" in outcome.unresolved(DAG, INC, w)
    assert "sp_fill_rate is failed" in outcome.explain(DAG, INC, w)


def test_a_rerun_without_the_patch_does_not(tmp_path):
    """Running broken code again runs the same broken code."""
    w = ran(tmp_path, [(RERUN, "sp_fill_rate"), (REFRESH, "fct_workforce"), (REFRESH, "rpt_workforce")])
    assert not outcome.resolved(DAG, INC, w)


def test_a_patch_nobody_ran_does_not(tmp_path):
    """Deploying a fix is not the same as the data being right: it has to be re-run afterwards."""
    w = ran(tmp_path, [(PATCH, "sp_fill_rate"), (REFRESH, "fct_workforce"), (REFRESH, "rpt_workforce")])
    assert not outcome.resolved(DAG, INC, w)


def test_the_right_fix_with_a_downstream_table_left_out_does_not(tmp_path):
    """The root cause is repaired and the report still shows the old numbers, because the report
    was never rebuilt. An approval review reads the plan; it rarely counts the refreshes."""
    w = ran(tmp_path, CORRECT[:-1])
    assert not outcome.resolved(DAG, INC, w)
    assert outcome.unresolved(DAG, INC, w) == ["rpt_workforce"]


def test_the_right_actions_in_the_wrong_order_do_not(tmp_path):
    """Every approved action happened exactly once. The report was rebuilt before the data behind
    it was fixed, so it was rebuilt from the broken numbers."""
    w = ran(tmp_path, [(REFRESH, "rpt_workforce"), (REFRESH, "fct_workforce"),
                       (PATCH, "sp_fill_rate"), (RERUN, "sp_fill_rate")])
    assert not outcome.resolved(DAG, INC, w)


def test_redoing_the_refresh_at_the_end_recovers_a_bad_order(tmp_path):
    """The order only matters for the last time a node is touched, which is why the fixer is free
    to choose it: a wasted early refresh is wasted, not wrong."""
    w = ran(tmp_path, [(REFRESH, "rpt_workforce")] + CORRECT)
    assert outcome.resolved(DAG, INC, w)


def test_an_untouched_branch_does_not_have_to_be_refreshed(tmp_path):
    """SYN-022 is the same break, and it marks fct_spend stale as well, because fct_spend sits under
    the broken report. Nothing on that branch broke: it is fed by load_vendors through
    sp_vendor_rollup, both fine. A plan that leaves it alone is complete, not incomplete."""
    inc = next(json.loads(l) for l in (DATA / "incidents.jsonl").read_text().splitlines() if '"SYN-022"' in l)
    assert inc["node_status"]["fct_spend"]["status"] == "stale"
    assert outcome.initial(DAG, inc)["fct_spend"] == outcome.FRESH
    assert outcome.resolved(DAG, inc, ran(tmp_path, CORRECT))


def test_a_missing_load_needs_only_a_run(tmp_path):
    """A load that never ran is not broken code: there is nothing to patch, and plan_to_calls does
    not emit a patch for one. It would be a bad model that demanded one."""
    inc = next(json.loads(l) for l in (DATA / "incidents.jsonl").read_text().splitlines() if '"SYN-002"' in l)
    assert inc["node_status"]["load_cases"]["status"] == "missing"
    w = ran(tmp_path, [(RERUN, "load_cases"), (RERUN, "sp_fill_rate"), (RERUN, "sp_merge_cases"),
                       (REFRESH, "fct_workforce"), (REFRESH, "rpt_workforce")])
    assert outcome.resolved(DAG, inc, w)


def test_doing_nothing_is_not_resolved(tmp_path):
    """If this ever passes, every test above is passing for the wrong reason."""
    w = ran(tmp_path, [])
    assert not outcome.resolved(DAG, INC, w)
    assert outcome.unresolved(DAG, INC, w)[0] == "rpt_workforce"


def test_the_action_log_is_ordered_across_all_three_tables(tmp_path):
    """The whole model depends on replaying actions in the order they happened, and they are
    recorded in three different tables."""
    w = ran(tmp_path, [(REFRESH, "rpt_workforce"), (PATCH, "sp_fill_rate"), (RERUN, "sp_fill_rate")])
    assert w.actions() == [(REFRESH, "rpt_workforce"), (PATCH, "sp_fill_rate"), (RERUN, "sp_fill_rate")]


def test_every_approved_plan_in_the_eval_set_resolves_its_incident(tmp_path):
    """The headline number: for all 100 labelled incidents, where the rules reach the approval gate,
    the plan a person is asked to approve is one that actually fixes the pipeline.

    This is the check that makes the metric worth reporting. If the model said yes to everything,
    the per-plan tests above would still pass and the number would mean nothing.
    """
    from triage_bridge.bench import run

    out = run(seed=5)
    n = out["reached_gate"]
    assert n >= 70
    assert out["resume"]["still_broken"] == []
    assert out["resume"]["symptom_resolved"] == f"{n}/{n}"
