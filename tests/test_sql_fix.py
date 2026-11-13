import shutil

import pytest

pytest.importorskip("duckdb")
pytest.importorskip("triage.warehouse")

from harness import Finish, RunStore, ScriptedModel, SimulatedCrash, ToolCall  # noqa: E402
from harness.types import DENIED  # noqa: E402
from triage.warehouse.pipeline import BY_ID  # noqa: E402
from triage_bridge import sql_fix  # noqa: E402
from triage_bridge.sql_fix import NeedsReview, NotApproved, Proposal  # noqa: E402
from triage_bridge.sql_warehouse import SqlWarehouse  # noqa: E402

BREAK, NODE, REPORT = "date_window.trailing_six_days", "rpt_revenue_dashboard", "rpt_revenue_dashboard"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    path = tmp_path_factory.mktemp("wh") / "broken.duckdb"
    SqlWarehouse.create(path, BREAK)
    return path


@pytest.fixture
def wh(built, tmp_path):
    shutil.copy(built, tmp_path / "wh.duckdb")
    return SqlWarehouse(tmp_path / "wh.duckdb")


def proposal(sql=None, reason="the trailing total summed six days"):
    return Proposal("CASE-1", REPORT, NODE, sql or BY_ID[NODE].sql, reason)


def crash(tool, event):
    fired = {"x": False}

    def hook(ev, idx, call):
        if not fired["x"] and ev == event and call.tool == tool:
            fired["x"] = True
            raise SimulatedCrash()
    return hook


def test_the_break_is_real_and_no_health_check_sees_it(wh):
    inc = wh.incident("CASE-1")
    assert inc["symptom"] == "trailing_7d_revenue in rpt_revenue_dashboard looks wrong"
    assert all(s["status"] == "ok" for s in inc["node_status"].values())


def test_nothing_is_applied_without_an_approval_for_this_exact_plan(wh):
    p, before = proposal(), wh.deployed(NODE)
    with pytest.raises(NotApproved):
        sql_fix.apply(wh, RunStore(), p, None)
    with pytest.raises(NotApproved):                       # approved, then the SQL changed
        sql_fix.apply(wh, RunStore(), proposal(p.sql.replace("- 7)", "- 30)")), p.plan_id)
    assert wh.deployed(NODE) == before and wh.counts()["patches"] == [] and wh.counts()["changes"] == {}


def test_an_approved_fix_is_deployed_run_and_verified(wh):
    p, store = proposal(), RunStore()
    run = sql_fix.apply(wh, store, p, p.plan_id)
    assert run["status"] == "done" and [s.split(" -> ")[1] for s in run["steps"]] == ["done", "done", "done"]
    out = sql_fix.verify_or_roll_back(wh, store, p)
    assert out["functional"] and out["integrity"] and out["kept"] and "rollback" not in out
    assert wh.deployed(NODE) == BY_ID[NODE].sql and wh.incident("CASE-1") is None
    assert wh.counts()["changes"] == {"CASE-1": 1} and wh.counts()["patches"] == [NODE]


def test_the_fixer_cannot_deploy_sql_a_person_did_not_approve(wh):
    p, store = proposal(), RunStore()
    own = p.sql.replace("- 7)", "- 30)")
    model = ScriptedModel([ToolCall("patch_procedure", {"node": NODE, "sql": own}),      # its own idea: refused
                           ToolCall("patch_procedure", {"node": "stg_orders"}),          # another node: refused
                           ToolCall("patch_procedure", {"node": NODE}),                  # sql left out: supplied
                           ToolCall("run_cycle", {}),
                           ToolCall("record_change", {"case_id": "CASE-1", "body": "window restored to seven days"}),
                           Finish("done")])
    sql_fix.apply(wh, store, p, p.plan_id, model=model)
    steps = store.steps("sqlfix-CASE-1")
    assert [bool(s["error"] and s["error"].startswith(DENIED)) for s in steps[:3]] == [True, True, False]
    assert "sql is fixed by the approval" in steps[0]["error"]
    assert wh.deployed(NODE) == p.sql and wh.deployed("stg_orders") == BY_ID["stg_orders"].sql
    assert wh.changes() == ["window restored to seven days"], "the wording is the model's, the SQL is not"


def test_a_crash_after_the_patch_is_resumed_without_repeating_anything(wh):
    p, store = proposal(), RunStore(wh.path + ".runs")
    with pytest.raises(SimulatedCrash):
        sql_fix.apply(wh, store, p, p.plan_id, fault_hook=crash("patch_procedure", "after_tool"))
    store.close()
    again = SqlWarehouse(wh.path)                                   # a new process: only the files survived
    store2 = RunStore(wh.path + ".runs")
    run = sql_fix.apply(again, store2, p, p.plan_id)
    assert run["status"] == "done" and again.counts()["patches"] == [NODE]
    assert again.counts()["changes"] == {"CASE-1": 1}
    assert sql_fix.verify_or_roll_back(again, store2, p)["kept"]
    store2.close()


def test_a_crash_while_recording_the_change_stops_for_a_person_and_records_it_once(wh):
    p, store = proposal(), RunStore(wh.path + ".runs")
    with pytest.raises(SimulatedCrash):
        sql_fix.apply(wh, store, p, p.plan_id, fault_hook=crash("record_change", "after_tool"))
    store.close()
    store2 = RunStore(wh.path + ".runs")
    with pytest.raises(NeedsReview) as stopped:
        sql_fix.apply(wh, store2, p, p.plan_id)
    step = stopped.value.step
    assert wh.took_effect(step["tool"], step["args"]), "a person looks, and it did land"
    store2.resolve_review(stopped.value.run_id, step["idx"], True, "checked the change log")
    assert sql_fix.apply(wh, store2, p, p.plan_id)["status"] == "done"
    assert wh.counts()["changes"] == {"CASE-1": 1}
    store2.close()


def test_an_approved_fix_that_does_not_verify_is_rolled_back_through_the_log(wh):
    """A person can approve a wrong fix. It is applied, because it was approved, and then undone."""
    before = wh.deployed(NODE)
    p, store = proposal(before.replace("- 6)", "- 5)"), "five looked right"), RunStore()
    assert sql_fix.apply(wh, store, p, p.plan_id)["status"] == "done" and wh.deployed(NODE) == p.sql
    out = sql_fix.verify_or_roll_back(wh, store, p)
    assert out["kept"] is False and out["rollback"]["status"] == "done"
    assert wh.deployed(NODE) == before, "exactly what was there before the fix"
    assert [s["tool"] for s in store.steps("rollback-CASE-1")] == ["patch_procedure", "run_cycle", "record_change"]
    assert wh.counts()["changes"] == {"CASE-1": 2} and "rolled back" in wh.changes()[-1]


def test_a_fix_that_only_brings_the_report_back_is_rolled_back(built, tmp_path):
    path = tmp_path / "join.duckdb"
    SqlWarehouse.create(path, "bad_join_key.part_without_supplier")
    wh = SqlWarehouse(path)
    report = BY_ID["rpt_regional_summary"]
    with wh.open() as raw:
        right = raw.db.execute("select margin_rate from reference.rpt_regional_summary limit 1").fetchone()[0]
    shortcut = report.sql.replace("(select sum(margin) / sum(revenue) from mart_supplier_performance)", f"{right!r}::double")
    p, store = Proposal("CASE-2", report.id, report.id, shortcut, "hard-coded the rate"), RunStore()
    sql_fix.apply(wh, store, p, p.plan_id)
    out = sql_fix.verify_or_roll_back(wh, store, p)
    assert out["functional"] is True and out["integrity"] is False and out["kept"] is False
    assert wh.deployed(report.id) == report.sql


def test_the_briefing_shows_the_change_and_the_plan_being_approved(wh):
    p = proposal()
    text = sql_fix.briefing(wh, p, "verified on a rehearsal copy")
    assert "- 6)" in text and "- 7)" in text and p.plan_id in text and "replace the SQL of rpt_revenue_dashboard" in text


def test_the_demo_runs_end_to_end_without_a_model(tmp_path, capsys):
    from triage_bridge import sql_demo

    out = sql_demo.main(["--draft", "reference", "--fixer", "replay", "--approve", "yes", "--crash",
                         "--workdir", str(tmp_path / "a")])
    assert out["result"] == "kept" and out["functional"] and out["integrity"] and out["changes"] == 1
    shown = capsys.readouterr().out
    assert "every node passes" in shown and "Process killed right after the patch" in shown and out["plan_id"] in shown

    no = sql_demo.main(["--draft", "reference", "--fixer", "replay", "--approve", "no", "--workdir", str(tmp_path / "b")])
    assert no["result"] == "rejected" and no["deployed_unchanged"]
