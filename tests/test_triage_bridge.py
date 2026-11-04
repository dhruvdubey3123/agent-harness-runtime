"""Triage running inside the harness. Skipped unless the [triage] extra is installed."""
import json

import pytest

pytest.importorskip("triage")
pytest.importorskip("langgraph.checkpoint.sqlite")

import openfiles  # noqa: E402  (sibling module: pytest puts tests/ on the path)

from harness import Finish, ScriptedModel, SimulatedCrash, ToolCall  # noqa: E402
from harness.types import DENIED  # noqa: E402
from triage_bridge import NeedsReview, TriageInHarness  # noqa: E402
from triage_bridge.runner import DATA  # noqa: E402

INC = next(json.loads(l) for l in (DATA / "incidents.jsonl").read_text().splitlines() if '"SYN-007"' in l)


def crash(tool, event):
    fired = {"x": False}

    def hook(ev, idx, call):
        if not fired["x"] and ev == event and call.tool == tool:
            fired["x"] = True
            raise SimulatedCrash()
    return hook


def test_nothing_runs_before_a_person_approves(tmp_path):
    t = TriageInHarness(tmp_path)
    v = t.diagnose(INC)
    assert v["waiting_for_human"] and t.apply_status("SYN-007") is None
    assert t.warehouse.counts()["job_runs"] == {}


def test_clean_apply_does_every_action_once(tmp_path):
    t = TriageInHarness(tmp_path)
    t.diagnose(INC)
    v = t.decide("SYN-007", True)
    c = t.warehouse.counts()
    assert v["outcome"] == "resolved" and c["job_runs"] == {"sp_fill_rate": 1} and c["comments"] == {"SYN-007": 1}
    assert t.warehouse.duplicated_actions() == 0


def test_reject_never_touches_the_warehouse(tmp_path):
    t = TriageInHarness(tmp_path)
    t.diagnose(INC)
    v = t.decide("SYN-007", False)
    assert v["outcome"] == "escalated" and t.apply_status("SYN-007") is None


def test_crash_after_rerun_stops_for_review_then_finishes_once(tmp_path):
    t = TriageInHarness(tmp_path, fault_hook=crash("rerun_job", "after_tool"))
    t.diagnose(INC)
    with pytest.raises(SimulatedCrash):
        t.decide("SYN-007", True)
    t2 = TriageInHarness(tmp_path)               # a fresh process: only SQLite survived
    with pytest.raises(NeedsReview):
        t2.recover("SYN-007")
    v = t2.resolve("SYN-007", t2.warehouse.took_effect("rerun_job", {"node": "sp_fill_rate"}))
    assert v["outcome"] == "resolved" and t2.warehouse.duplicated_actions() == 0
    assert t2.warehouse.counts()["job_runs"] == {"sp_fill_rate": 1}


def test_crash_before_a_side_effect_ran_is_retried_once(tmp_path):
    t = TriageInHarness(tmp_path, fault_hook=crash("comment_ticket", "during_tool"))
    t.diagnose(INC)
    with pytest.raises(SimulatedCrash):
        t.decide("SYN-007", True)
    t2 = TriageInHarness(tmp_path)
    with pytest.raises(NeedsReview):
        t2.recover("SYN-007")
    assert not t2.warehouse.took_effect("comment_ticket", {"case_id": "SYN-007"})
    t2.resolve("SYN-007", False)
    assert t2.warehouse.counts()["comments"] == {"SYN-007": 1}


def test_crash_on_idempotent_patch_resumes_without_a_person(tmp_path):
    t = TriageInHarness(tmp_path, fault_hook=crash("patch_procedure", "after_tool"))
    t.diagnose(INC)
    with pytest.raises(SimulatedCrash):
        t.decide("SYN-007", True)
    v = TriageInHarness(tmp_path).recover("SYN-007")
    assert v["outcome"] == "resolved"


@pytest.mark.skipif(not openfiles.available, reason="needs /proc to see open descriptors")
def test_the_benchmark_leaves_no_database_file_open():
    """Each incident runs in a temporary directory that is then deleted, which Windows refuses to
    do while a file is open. Three incidents is enough to catch a leak."""
    from triage_bridge.bench import run
    before = openfiles.open_db_files()
    run(seed=1, limit=3)
    assert openfiles.open_db_files() - before == set()


def test_bench_resume_never_duplicates():
    from triage_bridge.bench import run
    r = run(seed=3)
    assert r["resume"]["runs_with_duplicate"].startswith("0/")
    n = r["reached_gate"]
    assert r["resume"]["runs_exactly_once"] == f"{n}/{n}"


# ---------------------------------------------------- a fixer that decides for itself, against the
# real warehouse. Replaying the plan can only do what was approved; a model can try anything, so
# what holds it to the approval is the envelope in the sandbox rather than its instructions.

# What a model plausibly decides on its own: rerun the upstream load that caused the drift, and
# rerun the procedure once more to be sure. Both are refused; the wording is its own.
ROGUE = [
    ToolCall("patch_procedure", {"node": "sp_fill_rate", "change": "schema_drift"}),
    ToolCall("rerun_job", {"node": "load_cases"}),
    ToolCall("rerun_job", {"node": "sp_fill_rate"}),
    ToolCall("rerun_job", {"node": "sp_fill_rate"}),
    ToolCall("refresh_table", {"node": "fct_workforce"}),
    ToolCall("refresh_table", {"node": "rpt_workforce"}),
    ToolCall("comment_ticket", {"case_id": "SYN-007", "body": "Fixed. Wording chosen by the fixer."}),
    ToolCall("append_audit", {"line": "SYN-007: fixer wrote this line itself"}),
    Finish("Patched sp_fill_rate, reran it, refreshed both tables, commented and logged."),
]


def denials(status: dict) -> list[str]:
    return [s["error"] for s in status["steps"] if (s.get("error") or "").startswith(DENIED)]


def test_a_fixer_that_improvises_is_held_to_the_approved_plan(tmp_path):
    t = TriageInHarness(tmp_path, fixer=ScriptedModel(ROGUE))
    t.diagnose(INC)
    v = t.decide("SYN-007", True)

    counts = t.warehouse.counts()
    assert v["outcome"] == "resolved"
    assert counts["job_runs"] == {"sp_fill_rate": 1}          # load_cases never ran, sp_fill_rate ran once
    assert counts["comments"] == {"SYN-007": 1}
    assert t.warehouse.duplicated_actions() == 0

    refused = denials(t.apply_status("SYN-007"))
    assert len(refused) == 2
    assert any("not in the approved plan" in r for r in refused)
    assert any("already applied" in r for r in refused)


def test_the_words_are_the_fixers_own(tmp_path):
    """The approval fixes which node and which case, not the prose, which is the model's job."""
    t = TriageInHarness(tmp_path, fixer=ScriptedModel(ROGUE))
    t.diagnose(INC)
    t.decide("SYN-007", True)
    texts = t.warehouse.texts()
    assert texts["comments"] == ["Fixed. Wording chosen by the fixer."]
    assert texts["audit"] == ["SYN-007: fixer wrote this line itself"]


def test_the_approval_survives_a_crash_in_the_middle_of_a_model_run(tmp_path):
    """What is left of the approval is rebuilt from the run log, so a second process cannot
    re-spend an action the first one already used."""
    t = TriageInHarness(tmp_path, fault_hook=crash("rerun_job", "after_tool"), fixer=ScriptedModel(ROGUE))
    t.diagnose(INC)
    with pytest.raises(SimulatedCrash):
        t.decide("SYN-007", True)
    assert t.warehouse.counts()["job_runs"] == {"sp_fill_rate": 1}

    t2 = TriageInHarness(tmp_path, fixer=ScriptedModel(ROGUE))   # fresh process, only SQLite survived
    with pytest.raises(NeedsReview):
        t2.recover("SYN-007")
    v = t2.resolve("SYN-007", t2.warehouse.took_effect("rerun_job", {"node": "sp_fill_rate"}))
    assert v["outcome"] == "resolved"
    assert t2.warehouse.counts()["job_runs"] == {"sp_fill_rate": 1}
    assert t2.warehouse.duplicated_actions() == 0
