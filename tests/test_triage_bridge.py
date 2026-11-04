"""Triage running inside the harness. Skipped unless the [triage] extra is installed."""
import json

import pytest

pytest.importorskip("triage")
pytest.importorskip("langgraph.checkpoint.sqlite")

from harness import SimulatedCrash  # noqa: E402
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


def test_bench_resume_never_duplicates():
    from triage_bridge.bench import run
    r = run(seed=3)
    assert r["resume"]["runs_with_duplicate"].startswith("0/")
    n = r["reached_gate"]
    assert r["resume"]["runs_exactly_once"] == f"{n}/{n}"
