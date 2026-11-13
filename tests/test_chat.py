"""The chat is a front door on the same flow as the terminal demo. These run it with no model."""
import pytest

pytest.importorskip("triage.warehouse")
from fastapi.testclient import TestClient

from triage_bridge.chat import Session, build_app, diff, readable

BREAK, NODE, REPORT = "type_drift.price_as_integer", "stg_lineitem", "rpt_revenue_dashboard"


@pytest.fixture
def chat(tmp_path):
    def make(**kw):
        s = Session(tmp_path, BREAK, draft="reference", fixer="replay", background=False, **kw)
        return s, TestClient(build_app(s))
    return make


def kinds(c):
    return [e["kind"] for e in c.get("/api/events").json()["events"]]


def proposal(c):
    return next(e for e in c.get("/api/events").json()["events"] if e["kind"] == "proposal")


def test_a_complaint_ends_with_a_plan_waiting_and_nothing_applied(chat):
    s, c = chat()
    broken = s.wh.deployed(NODE)
    assert c.post("/api/complain", json={"text": "revenue looks off", "report": REPORT}).json()["state"] == "awaiting"
    p = proposal(c)
    assert p["node"] == NODE and p["rehearsal"].startswith("verified")
    assert any(l.startswith("-") for l in p["diff"]) and any(l.startswith("+") for l in p["diff"])
    assert s.wh.deployed(NODE) == broken and s.wh.counts()["changes"] == {}


def test_approving_the_plan_fixes_the_dashboard_and_records_it_once(chat):
    s, c = chat()
    before = c.get(f"/api/report/{REPORT}").json()["rows"]
    c.post("/api/complain", json={"text": "revenue looks off", "report": REPORT})
    assert c.post("/api/decide", json={"plan_id": proposal(c)["plan_id"], "approve": True}).json()["state"] == "idle"
    r = next(e for e in c.get("/api/events").json()["events"] if e["kind"] == "result")
    assert r["kept"] and r["functional"] and r["integrity"] and r["changes"] == 1
    assert c.get(f"/api/report/{REPORT}").json()["rows"] != before


def test_rejecting_applies_nothing(chat):
    s, c = chat()
    broken, before = s.wh.deployed(NODE), c.get(f"/api/report/{REPORT}").json()
    c.post("/api/complain", json={"text": "revenue looks off", "report": REPORT})
    c.post("/api/decide", json={"plan_id": proposal(c)["plan_id"], "approve": False})
    assert s.wh.deployed(NODE) == broken and s.wh.counts()["changes"] == {}
    assert c.get(f"/api/report/{REPORT}").json() == before and "result" not in kinds(c)


def test_only_the_plan_on_the_table_can_be_approved(chat):
    s, c = chat()
    broken = s.wh.deployed(NODE)
    assert c.post("/api/decide", json={"plan_id": "abc", "approve": True}).status_code == 409      # nothing waiting
    c.post("/api/complain", json={"text": "revenue looks off", "report": REPORT})
    assert c.post("/api/decide", json={"plan_id": "not-the-plan", "approve": True}).status_code == 409
    assert s.wh.deployed(NODE) == broken and s.state == "awaiting"
    assert c.post("/api/complain", json={"text": "and another thing", "report": REPORT}).status_code == 409


def test_a_killed_apply_resumes_and_records_the_change_once(chat):
    s, c = chat(crash=True)
    c.post("/api/complain", json={"text": "revenue looks off", "report": REPORT})
    c.post("/api/decide", json={"plan_id": proposal(c)["plan_id"], "approve": True})
    ev = c.get("/api/events").json()["events"]
    assert any(e.get("killed") for e in ev)
    r = next(e for e in ev if e["kind"] == "result")
    assert r["kept"] and r["changes"] == 1 and s.wh.counts()["patches"].count(NODE) == 1


def test_once_fixed_there_is_nothing_left_to_trace(chat):
    s, c = chat()
    c.post("/api/complain", json={"text": "revenue looks off", "report": REPORT})
    c.post("/api/decide", json={"plan_id": proposal(c)["plan_id"], "approve": True})
    c.post("/api/complain", json={"text": "still wrong?", "report": REPORT})
    assert kinds(c)[-1] == "agent" and "proposal" not in kinds(c)[kinds(c).index("result"):]


def test_the_page_and_the_key_stay_apart(chat, monkeypatch):
    monkeypatch.setenv("HARNESS_API_KEY", "sk-secret-value")
    s, c = chat()
    assert "sk-secret-value" not in c.get("/").text + c.get("/api/meta").text
    assert c.get("/api/meta").json()["mode"] == "scripted, no model"


def test_the_diff_is_readable_and_the_stored_sql_is_untouched():
    old = "select a as x, cast(b as integer) as y, coalesce(c, 0) as z from t where a > 1"
    new = old.replace("cast(b as integer)", "b")
    assert readable(old) == ["select a as x,", "cast(b as integer) as y,", "coalesce(c, 0) as z", "from t", "where a > 1"]
    assert [l for l in diff(old, new) if l[0] in "+-"] == ["-cast(b as integer) as y,", "+b as y,"]
