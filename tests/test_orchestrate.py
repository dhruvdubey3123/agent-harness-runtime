"""Two agents and a person, driven by a stub model so CI can run it.

What is worth asserting here is the shape of the sequence rather than the quality of the answer:
the reasoning agent cannot touch the warehouse, a malformed diagnosis comes back to it as an error
rather than reaching a person, the diagnosis a person approves is the one that was submitted, and a
crash between the two agents loses neither.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("triage")

from harness import OpenAICompatModel, SimulatedCrash  # noqa: E402
from harness.types import DENIED  # noqa: E402
from stub_model_server import serving  # noqa: E402  (sibling module: pytest puts tests/ on the path)
from triage_bridge.orchestrate import Escalated, TwoAgentTriage  # noqa: E402
from triage_bridge.runner import DATA, NeedsReview  # noqa: E402

INC = next(json.loads(line) for line in (DATA / "incidents.jsonl").read_text().splitlines() if '"SYN-007"' in line)
CASE = "SYN-007"


@pytest.fixture
def model(monkeypatch):
    with serving() as url:
        monkeypatch.setenv("NO_PROXY", "*")
        yield OpenAICompatModel(model="stub", base_url=url)


def errors(steps: list[dict]) -> list[str]:
    return [s["error"] for s in steps if s.get("error")]


def test_the_model_diagnoses_a_person_approves_and_the_model_applies(model, tmp_path):
    t = TwoAgentTriage(tmp_path, model=model)
    v = t.diagnose(INC)
    assert v["waiting_for_human"] and v["review_issues"] == []
    assert v["diagnosis"]["fix_target"] == "sp_fill_rate" and v["diagnosis"]["category"] == "schema_drift"
    assert v["diagnosis"]["plan"] == ["fix sp_fill_rate", "refresh fct_workforce", "refresh rpt_workforce"]

    v = t.decide(CASE, True)
    counts = t.warehouse.counts()
    assert v["outcome"] == "resolved"
    assert counts["job_runs"] == {"sp_fill_rate": 1} and counts["comments"] == {CASE: 1}
    assert t.warehouse.duplicated_actions() == 0
    assert [line for line in t.trace(CASE) if line.startswith("[")] == ["[diagnose] done", "[apply] done"]


def test_the_reasoning_agent_has_no_tool_that_changes_anything(model, tmp_path):
    """It asks for one anyway, which is the point: the refusal comes from its sandbox, not its prompt."""
    t = TwoAgentTriage(tmp_path, model=model)
    t.diagnose(INC)
    assert t.warehouse.counts()["job_runs"] == {}
    refused = [e for e in errors(t.store.steps(f"diag-{CASE}")) if e.startswith(DENIED)]
    assert any("rerun_job" in e and "not allowed for this agent" in e for e in refused)


def test_a_malformed_diagnosis_goes_back_to_the_model_and_not_to_a_person(model, tmp_path):
    t = TwoAgentTriage(tmp_path, model=model)
    t.diagnose(INC)
    steps = t.store.steps(f"diag-{CASE}")
    submits = [s for s in steps if s["tool"] == "submit_diagnosis"]
    assert len(submits) == 2
    assert submits[0]["status"] == "error" and "is not a category" in submits[0]["error"]
    assert submits[1]["status"] == "done"
    assert t.diagnosis(CASE).category == "schema_drift"   # the valid one is what a person sees


def test_the_diagnosis_comes_from_the_log_not_from_the_process_that_made_it(model, tmp_path):
    TwoAgentTriage(tmp_path, model=model).diagnose(INC)
    fresh = TwoAgentTriage(tmp_path, model=model)          # new process, only SQLite survived
    d = fresh.diagnosis(CASE)
    assert d is not None and d.fix_target == "sp_fill_rate" and d.cause == "load_cases"
    assert fresh.view(CASE)["waiting_for_human"]


def test_a_crash_in_the_diagnosis_is_picked_up_by_the_next_process(model, tmp_path):
    fired = {"x": False}

    def hook(event, idx, call):
        if event == "after_tool" and call.tool == "precedents" and not fired["x"]:
            fired["x"] = True
            raise SimulatedCrash()

    t = TwoAgentTriage(tmp_path, model=model, fault_hook=hook)
    with pytest.raises(SimulatedCrash):
        t.diagnose(INC)
    assert t.diagnosis(CASE) is None                       # it had not submitted yet

    v = TwoAgentTriage(tmp_path, model=model).recover(CASE)
    assert v["diagnosis"]["fix_target"] == "sp_fill_rate" and v["waiting_for_human"]


def test_a_crash_in_the_fix_after_a_model_diagnosis_still_applies_each_action_once(model, tmp_path):
    fired = {"x": False}

    def hook(event, idx, call):
        if event == "after_tool" and call.tool == "rerun_job" and not fired["x"]:
            fired["x"] = True
            raise SimulatedCrash()

    t = TwoAgentTriage(tmp_path, model=model, fault_hook=hook)
    t.diagnose(INC)
    with pytest.raises(SimulatedCrash):
        t.decide(CASE, True)

    t2 = TwoAgentTriage(tmp_path, model=model)
    with pytest.raises(NeedsReview):
        t2.recover(CASE)
    v = t2.resolve(CASE, t2.warehouse.took_effect("rerun_job", {"node": "sp_fill_rate"}))
    assert v["outcome"] == "resolved"
    assert t2.warehouse.counts()["job_runs"] == {"sp_fill_rate": 1}
    assert t2.warehouse.duplicated_actions() == 0


def test_a_rejected_diagnosis_never_reaches_the_warehouse(model, tmp_path):
    t = TwoAgentTriage(tmp_path, model=model)
    t.diagnose(INC)
    v = t.decide(CASE, False)
    assert v["outcome"] == "escalated" and t.warehouse.counts()["job_runs"] == {}
    with pytest.raises(Escalated):
        t.apply(CASE)


# ------------------------------------------------------------------ scoring the first agent

def test_the_rule_based_baseline_still_matches_every_label():
    """The number the model is measured against. It is in the README, so it is asserted here."""
    from triage_bridge.diag_bench import incidents, rules, tally
    r = rules(incidents())
    assert tally(r["cases"], "root_cause_ok") == "100/100"
    assert tally(r["cases"], "category_ok") == "93/100"


def test_the_diagnosis_benchmark_scores_both_paths(model, tmp_path):
    from triage_bridge.diag_bench import run
    res = run(model=model, only=[CASE], workdir=tmp_path)
    assert res["model_diagnoser"]["root cause correct"] == "1/1"
    assert res["model_diagnoser"]["category correct"] == "1/1"
    assert res["rule_based_tracer"]["root cause correct"] == "1/1"
    assert res["raw"]["model"][0]["cause"] == "load_cases"


# ------------------------------------------------------------------ the tracer checks the model's work

def scripted(*calls):
    from harness import Finish, ScriptedModel, ToolCall
    return ScriptedModel([ToolCall(n, a) for n, a in calls] + [Finish("submitted")])


def test_a_diagnosis_the_tracer_disagrees_with_never_reaches_approval(tmp_path):
    wrong = {"category": "schema_drift", "fix_target": "load_cases", "reason": "load_cases changed a column."}
    t = TwoAgentTriage(tmp_path, model=scripted(("submit_diagnosis", wrong)))
    v = t.diagnose(INC)
    assert not v["waiting_for_human"] and any("disagree on the root cause" in i for i in v["review_issues"])
    with pytest.raises(Escalated):
        t.decide(CASE, True)
    assert t.warehouse.counts()["job_runs"] == {}


def test_a_failure_the_rules_do_not_recognise_goes_to_a_person_whatever_the_model_calls_it(tmp_path):
    """Regression from a real Qwen3 8B run in pipeline-triage-graph: on failures the rules call
    'unknown', the model named a familiar category and the incident looked routine."""
    import copy

    from triage.dag import PipelineDAG
    from triage.knowledge import CaseLibrary
    from triage.service import TriageService

    rows = [json.loads(line) for line in (DATA / "incidents.jsonl").read_text().splitlines()]
    inc = next(r for r in rows if r["id"] == "SYN-043")
    bare = {k: v for k, v in inc.items() if k != "label"}
    lib = CaseLibrary.load(DATA / "case_library.yaml")
    right = TriageService(PipelineDAG.from_yaml(DATA / "pipeline.yaml"), CaseLibrary(copy.deepcopy(lib.cases))
                          ).open(bare)[1]["fix_target"]

    t = TwoAgentTriage(tmp_path, model=scripted(("submit_diagnosis", {"category": "schema_drift",
                                                                       "fix_target": right, "reason": "x"})))
    v = t.diagnose(inc)
    assert v["diagnosis"]["fix_target"] == right                     # right place
    assert "unrecognised failure category" in v["review_issues"]     # and still a person's call
    assert not v["waiting_for_human"]
