"""The chat is a front door on the same flow as the terminal demo, with tickets arriving night by
night. These run it with no model: the scripted drafter reads the answer off the warehouse, so what
is tested is the flow and its guarantees, not anyone's judgement."""
import pytest

pytest.importorskip("triage.warehouse.world")
from fastapi.testclient import TestClient

from triage_bridge.chat import Session, build_app, diff, readable

BREAK, NODE, REPORT = "type_drift.price_as_integer", "stg_lineitem", "rpt_revenue_dashboard"


@pytest.fixture
def chat(tmp_path):
    def make(break_id=BREAK, **kw):
        s = Session(tmp_path, break_id, draft="reference", fixer="replay", background=False, **kw)
        return s, TestClient(build_app(s))
    return make


def state(c):
    return c.get("/api/events").json()


def last(c, kind):
    return [e for e in state(c)["events"] if e["kind"] == kind][-1]


def start(c):
    """Run a night and hand its ticket to the agent. -> the ticket id."""
    t = c.post("/api/night").json()["ticket"]
    c.post("/api/work", json={"ticket": t})
    return t


def test_a_ticket_ends_with_a_plan_waiting_and_nothing_applied(chat):
    s, c = chat()
    assert state(c)["tickets"] == []
    t = start(c)
    assert state(c)["state"] == "awaiting" and state(c)["tickets"][0]["status"] == "waiting for your approval"
    p = last(c, "proposal")
    assert p["ticket"] == t and p["node"] == NODE and p["rehearsal"].startswith("verified")
    assert any(l.startswith("-") for l in p["diff"]) and any(l.startswith("+") for l in p["diff"])
    assert "cast(l_extendedprice as integer)" in s.wh.deployed(NODE) and s.wh.counts()["changes"] == {}
    assert "was" not in state(c)["tickets"][0]            # what it really was is not shown while it is open


def test_approving_fixes_the_dashboard_records_it_once_and_closes_the_ticket(chat):
    s, c = chat()
    t = start(c)
    before = c.get(f"/api/report/{REPORT}").json()["rows"]
    assert c.post("/api/decide", json={"plan_id": last(c, "proposal")["plan_id"], "approve": True}).json()["state"] == "idle"
    r = last(c, "result")
    assert r["kept"] and r["functional"] and r["integrity"] and r["changes"] == 1
    assert c.get(f"/api/report/{REPORT}").json()["rows"] != before
    ticket = state(c)["tickets"][0]
    assert ticket["status"] == "closed" and ticket["right"] and NODE in ticket["was"]
    assert state(c)["tally"] == {"night": 1, "closed": 1, "by_agent": 1}


def test_rejecting_applies_nothing_and_the_ticket_goes_to_a_person(chat):
    s, c = chat()
    t = start(c)
    broken, before = s.wh.deployed(NODE), c.get(f"/api/report/{REPORT}").json()
    c.post("/api/decide", json={"plan_id": last(c, "proposal")["plan_id"], "approve": False})
    assert s.wh.deployed(NODE) == broken and s.wh.counts()["changes"] == {}
    assert c.get(f"/api/report/{REPORT}").json() == before
    assert state(c)["tickets"][0]["status"] == "needs a person"
    assert c.post("/api/night").status_code == 409                          # the next night waits for the open ticket
    c.post("/api/resolve", json={"ticket": t, "action": "person"})
    ticket = state(c)["tickets"][0]
    assert ticket["status"] == "closed" and not ticket["right"] and ticket["outcome"] == "fixed by a person"
    assert c.post("/api/night").status_code == 200


def test_only_the_plan_on_the_table_can_be_approved(chat):
    s, c = chat()
    assert c.post("/api/decide", json={"plan_id": "abc", "approve": True}).status_code == 409      # nothing waiting
    start(c)
    broken = s.wh.deployed(NODE)
    assert c.post("/api/decide", json={"plan_id": "not-the-plan", "approve": True}).status_code == 409
    assert s.wh.deployed(NODE) == broken and s.state == "awaiting"
    assert c.post("/api/complain", json={"text": "and another thing", "report": REPORT}).status_code == 409


def test_a_killed_apply_resumes_and_records_the_change_once(chat):
    s, c = chat(crash=True)
    t = start(c)
    c.post("/api/decide", json={"plan_id": last(c, "proposal")["plan_id"], "approve": True})
    ev = state(c)["events"]
    assert any(e.get("killed") for e in ev)
    assert last(c, "result")["kept"] and s.wh.counts()["changes"] == {t: 1} and s.wh.counts()["patches"].count(NODE) == 1


def nights_until(s, c, kind):
    """Play nights with the scripted drafter until a ticket of this kind is open and with the agent."""
    for _ in range(40):
        t = c.post("/api/night").json()["ticket"]
        if t is None:
            continue
        ticket = next(x for x in s.tickets if x["id"] == t)
        c.post("/api/work", json={"ticket": t})
        if ticket["case"]["kind"] == kind:
            return ticket
        if s.state == "awaiting":
            c.post("/api/decide", json={"plan_id": s.proposal.plan_id, "approve": True})
        else:
            c.post("/api/resolve", json={"ticket": t, "action": "reload" if ticket["agent"]["closed"]["kind"] == "source_data" else "accept"})
        assert ticket["status"] == "closed", ticket
    raise AssertionError(f"no {kind} ticket in 40 nights")


def test_bad_source_data_ends_with_the_table_named_and_no_sql_changed(chat):
    s, c = chat(break_id=None)
    ticket = nights_until(s, c, "source")
    closure = last(c, "closure")
    assert closure["ending"] == "source_data" and closure["table"] == ticket["case"]["table"]
    assert ticket["status"] == "your call" and "proposal" not in [e["kind"] for e in state(c)["events"] if e["ticket"] == ticket["id"]]
    with s.wh.open() as raw:
        assert not raw.changed_procedures() and raw.affected()              # nothing patched, and it is still wrong
    assert c.post("/api/resolve", json={"ticket": ticket["id"], "action": "accept"}).status_code == 409
    c.post("/api/resolve", json={"ticket": ticket["id"], "action": "reload"})
    assert ticket["status"] == "closed" and ticket["right"] and ticket["case"]["table"] in ticket["was"]
    with s.wh.open() as raw:
        assert not raw.affected()


def test_a_wrong_ending_does_not_close_the_ticket(chat):
    """The agent blames the wrong source. Its owner reloads it, nothing improves, and the check says so."""
    s, c = chat(break_id=None)
    ticket = nights_until(s, c, "source")
    wrong = next(x for x in ("orders", "customer", "supplier") if x != ticket["case"]["table"])
    ticket["agent"]["closed"]["source_table"] = wrong
    c.post("/api/resolve", json={"ticket": ticket["id"], "action": "reload"})
    assert ticket["status"] == "needs a person" and "still wrong" in ticket["outcome"]
    assert last(c, "verdict")["ok"] is False and "was" not in state(c)["tickets"][-1]
    c.post("/api/resolve", json={"ticket": ticket["id"], "action": "person"})
    assert ticket["status"] == "closed" and not ticket["right"]


def test_a_mistaken_complaint_ends_with_nothing_changed(chat):
    s, c = chat(break_id=None)
    ticket = nights_until(s, c, "none")
    assert last(c, "closure")["ending"] == "nothing_wrong"
    c.post("/api/resolve", json={"ticket": ticket["id"], "action": "accept"})
    assert ticket["status"] == "closed" and ticket["right"] and ticket["was"].startswith("nothing")
    assert s.wh.counts()["patches"] == [] or NODE not in s.wh.counts()["patches"]


def test_a_typed_complaint_is_a_ticket_like_any_other(chat):
    s, c = chat(break_id=None)
    assert c.post("/api/complain", json={"text": "revenue looks off", "report": REPORT}).status_code == 200
    ticket = state(c)["tickets"][0]
    assert ticket["raised_by"] == "you" and ticket["status"] == "your call" and last(c, "closure")["ending"] == "nothing_wrong"


def test_a_quiet_night_raises_no_ticket(chat):
    s, c = chat(break_id=None)
    s.plan[0] = None
    assert c.post("/api/night").json()["ticket"] is None
    assert state(c)["tickets"] == [] and last(c, "night")["text"].endswith("No tickets.")


def test_the_page_and_the_key_stay_apart(chat, monkeypatch):
    monkeypatch.setenv("HARNESS_API_KEY", "sk-secret-value")
    s, c = chat()
    assert "sk-secret-value" not in c.get("/").text + c.get("/api/meta").text
    assert c.get("/api/meta").json()["mode"] == "scripted, no model"


def test_the_diff_is_readable_and_the_stored_sql_is_untouched():
    old = "select a as x, cast(b as integer) as y, coalesce(c, 0) as z from t left join u on u.k = t.k where a > 1"
    new = old.replace("cast(b as integer)", "b")
    assert readable(old) == ["select a as x,", "cast(b as integer) as y,", "coalesce(c, 0) as z", "from t", "left join u on u.k = t.k", "where a > 1"]
    assert [l for l in diff(old, new) if l[0] in "+-"] == ["-cast(b as integer) as y,", "+b as y,"]
