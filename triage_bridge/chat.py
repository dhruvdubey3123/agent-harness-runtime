"""The analyst's side of it: a dashboard, a queue of tickets that fills up night by night, and a chat.

    python -m triage_bridge.chat                                   # a model drafts and applies
    python -m triage_bridge.chat --draft reference --fixer replay   # no model, no key
    python -m triage_bridge.chat --crash                            # kill each apply after the patch, then resume
    python -m triage_bridge.chat --history                          # the agent may read what was deployed
    python -m triage_bridge.chat --break type_drift.price_as_integer    # plant this one on the first night

Then open http://127.0.0.1:8765 and run the next night. Each night the pipeline rebuilds, and on
some nights something has gone wrong: a colleague's change to one procedure, a source table that
arrived bad, or nothing at all and an analyst who is mistaken. A ticket appears. Nobody is told which
kind it is, the agent included.

The agent takes one ticket at a time and ends it one of three ways. A fix, which you approve by plan
id and Keel applies through its run log. A bad source, which it names and does not paper over; the
owner reloads it. Or nothing wrong. After each, the warehouse is checked against its reference, and
only then is the ticket shown what it really was. A ticket the agent got wrong stays open for a
person.

It does one thing: a complaint about a report, to its right ending. It is not a general assistant
for the warehouse. The data is a frozen sample, so a night brings a new fault, not new rows.

Models are read from HARNESS_MODEL / HARNESS_BASE_URL / HARNESS_API_KEY, falling back to the TRIAGE_
variables. The key is only ever read from the environment and never sent to the page.
"""
from __future__ import annotations

import argparse
import difflib
import re
import tempfile
import threading
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from triage.warehouse import world
from triage.warehouse.breaks import BREAKS
from triage.warehouse.pipeline import REPORTS
from triage.warehouse.probe import observe, observe_sources

from harness import OpenAICompatModel, RunStore, SimulatedCrash

from . import sql_demo, sql_fix
from .sql_fix import Incomplete, NeedsReview, Proposal
from .sql_warehouse import SqlWarehouse

PAGE = Path(__file__).parent / "data" / "chat.html"
BREAKS_AT = re.compile(r"(?<!left)(?<!inner)\s+(?=(?:from|where|group by|order by|left join|inner join|join|having|union)\b)", re.I)
WAS = {"sql": "a change to the SQL of {node}: {what}", "source": "bad source data in raw.{table}: {what}",
       "none": "nothing: the report was right"}


def readable(sql: str) -> list[str]:
    """SQL is stored as one line. For a person to read a diff of it, break it at commas and clauses.
    This is for display only: the plan id is a hash of the SQL as stored."""
    out = []
    for part in BREAKS_AT.split(sql.strip()):
        out += [s.strip() for s in re.sub(r",\s+(?![^()]*\))", ",\n", part).split("\n") if s.strip()]
    return out


def diff(old: str, new: str) -> list[str]:
    lines = difflib.unified_diff(readable(old), readable(new), lineterm="", n=1)
    return [l for l in lines if not l.startswith(("---", "+++", "@@"))]


def _plain(v):
    return float(v) if isinstance(v, Decimal) else v if isinstance(v, (int, float, str, type(None))) else str(v)


class Session:
    """One warehouse, a queue of tickets, one conversation. The agent's state moves
    idle -> working -> awaiting -> applying -> idle, on one ticket at a time. Everything the page
    shows is an event in `events`; the page polls for the ones it has not seen."""

    def __init__(self, workdir: str | Path, break_id: str | None = None, draft: str = "model", fixer: str = "model",
                 crash: bool = False, background: bool = True, seed: int = 0, history: bool = False):
        self.dir = Path(workdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.break_id, self.draft, self.fixer, self.crash, self.background = break_id, draft, fixer, crash, background
        self.seed, self.history = seed, history
        self.lock = threading.Lock()
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        with self.lock:
            for f in self.dir.glob("*.d*b*"):
                f.unlink()
            self.wh = SqlWarehouse.create(self.dir / "warehouse.duckdb")
            with self.wh.open() as raw:
                cases = world.cases(raw)
                raw.run_cycle()
            self.plan = world.schedule(cases, 60, self.seed)
            if self.break_id:
                b = next(x for x in BREAKS if x.id == self.break_id)
                self.plan.insert(0, {"id": b.id, "kind": "sql", "node": b.node, "sql": b.broken_sql(), "what": b.what})
            self.events: list[dict] = []
            self.tickets: list[dict] = []
            self.state, self.proposal, self.active, self.night = "idle", None, None, 0
            self.reports = self._read_reports()

    def _read_reports(self) -> dict:
        out = {}
        with self.wh.open() as raw:
            for r in REPORTS:
                cur = raw.db.execute(f"select * from {r} order by 1, 2")
                cols = [d[0] for d in cur.description]
                out[r] = {"columns": cols, "rows": [[_plain(v) for v in row] for row in cur.fetchall()],
                          "refreshed": raw.last_run(r)["status"] == "success"}
        return out

    def say(self, kind: str, **body) -> None:
        self.events.append({"n": len(self.events), "kind": kind, "ticket": self.active["id"] if self.active else None, **body})

    def _go(self, fn, *args) -> None:
        def guarded():
            try:
                fn(*args)
            except BaseException as e:      # a model with no key raises SystemExit; the page must still hear about it
                self.say("agent", text=f"I had to stop: {e}. Nothing was applied beyond what the log above shows.")
                self._needs_person("the agent stopped with an error")
        if self.background:
            threading.Thread(target=guarded, daemon=True).start()
        else:
            guarded()

    def view(self) -> list[dict]:
        """Tickets as the page may see them: what a ticket really was is shown only once it is closed."""
        keep = ("id", "night", "report", "text", "raised_by", "status", "outcome")
        return [{**{k: t[k] for k in keep}, **({"was": t["was"], "right": t["right"]} if t["status"] == "closed" else {})}
                for t in self.tickets]

    def _open(self) -> dict | None:
        return next((t for t in self.tickets if t["status"] != "closed"), None)

    def _ticket(self, ticket_id: str, *statuses: str) -> dict:
        t = next((x for x in self.tickets if x["id"] == ticket_id), None)
        if t is None:
            raise HTTPException(404, f"no ticket {ticket_id}")
        if statuses and t["status"] not in statuses:
            raise HTTPException(409, f"{ticket_id} is {t['status']}")
        return t

    def _busy(self) -> None:
        if self.state != "idle":
            raise HTTPException(409, f"I'm {self.state.replace('awaiting', 'waiting for your decision')}; one ticket at a time.")

    # ------------------------------------------------------------------ what a person can do
    def next_night(self) -> dict | None:
        """The pipeline rebuilds. Sometimes a ticket comes of it. -> the ticket, or None for a quiet night."""
        self._busy()
        if self._open():
            raise HTTPException(409, f"{self._open()['id']} is still open. Close it before the next night runs.")
        self.night += 1
        if self.night > len(self.plan):
            with self.wh.open() as raw:
                self.plan += world.schedule(world.cases(raw), 60, self.seed + self.night)
        case = self.plan[self.night - 1]
        self.active = None
        if case is None:
            with self.wh.open() as raw:
                raw.mark_good()
                raw.run_cycle()
            self.say("night", night=self.night, text=f"Night {self.night}: all twelve tables rebuilt. No tickets.")
            return None
        with self.wh.open() as raw:
            incident = world.stage(raw, case)
            seen = any(s["status"] != "ok" for s in incident["node_status"].values()) or bool(observe_sources(raw))
        t = self._file(incident["symptom_node"], incident["symptom"], "a failed health check, then an analyst" if seen else "an analyst",
                       case, incident["node_status"])
        self.say("night", night=self.night, text=f"Night {self.night}: all twelve tables rebuilt. Ticket {t['id']} was raised.")
        with self.lock:
            self.reports = self._read_reports()
        return t

    def _file(self, report: str, text: str, raised_by: str, case: dict, node_status: dict) -> dict:
        t = {"id": f"T-{len(self.tickets) + 1}", "night": self.night, "report": report, "text": text, "raised_by": raised_by,
             "status": "open", "outcome": "", "case": case, "node_status": node_status, "agent": None}
        self.tickets.append(t)
        return t

    def work(self, ticket_id: str) -> None:
        self._busy()
        t = self._ticket(ticket_id, "open")
        self.state, self.active, t["status"] = "working", t, "with the agent"
        self.say("user", text=f"{t['id']}: {t['text']}")
        self._go(self._investigate, t)

    def complain(self, text: str, report: str) -> None:
        """An analyst types a complaint of their own. It is a ticket like any other."""
        self._busy()
        if report not in REPORTS:
            raise HTTPException(404, f"no report called {report}")
        if self._open():
            raise HTTPException(409, f"{self._open()['id']} is still open. Work that one first.")
        with self.wh.open() as raw:
            status = observe(raw)
        t = self._file(report, f"an analyst looking at {report} says: {text}", "you", {"id": "typed", "kind": "none", "what": "nothing is wrong"}, status)
        self.work(t["id"])

    def decide(self, plan_id: str, approve: bool) -> None:
        p, t = self.proposal, self.active
        if self.state != "awaiting" or p is None:
            raise HTTPException(409, "there is no plan waiting for a decision")
        if plan_id != p.plan_id:
            raise HTTPException(409, f"plan {plan_id} is not the plan on the table ({p.plan_id})")
        if not approve:
            self.say("user", text=f"Reject plan {p.plan_id}")
            self.say("agent", text="Rejected. Nothing was applied, and the warehouse is as it was.")
            self._needs_person("you rejected the plan")
            return
        self.state = "applying"
        self.say("user", text=f"Approve plan {p.plan_id}")
        self._go(self._apply, p, t)

    def resolve(self, ticket_id: str, action: str) -> None:
        """What happens outside the agent: the source's owner reloads a table, a person accepts that
        nothing is wrong, or the on-call engineer takes the ticket and fixes it by hand."""
        self._busy()
        t = self._ticket(ticket_id, "your call", "needs a person")
        self.active = t
        closed = (t["agent"] or {}).get("closed") or {}
        if action == "reload" and t["status"] == "your call" and closed.get("kind") == "source_data":
            with self.wh.open() as raw:
                done = raw.reload_sources(only=closed["source_table"])
                raw.run_cycle()
            self.say("note", text=f"The owner of raw.{closed['source_table']} reloaded it"
                     + ("" if done else ", but it was not the table that was bad") + ". The pipeline was rebuilt.")
            self._settle(t, f"raw.{closed['source_table']} reloaded by its owner")
        elif action == "accept" and t["status"] == "your call" and closed.get("kind") == "nothing_wrong":
            self._settle(t, "closed: no fault found")
        elif action == "person":
            with self.wh.open() as raw:
                world.restore(raw)
            self.say("note", text="The on-call engineer took the ticket and put the warehouse right by hand.")
            self._settle(t, "fixed by a person", by_person=True)
        else:
            raise HTTPException(409, f"'{action}' is not available for {ticket_id}")

    # ------------------------------------------------------------------ the work
    def _investigate(self, t: dict) -> None:
        incident = {"id": t["id"], "symptom_node": t["report"], "symptom": t["text"], "node_status": t["node_status"]}
        self.say("agent", text=f"Looking into {t['report']}. While I investigate I can read tables and their SQL, and I can't change anything.")
        d = sql_demo.draft_run(self.wh, incident, self.draft, history=self.history, open_world=True)
        for s in d["steps"]:
            self.say("tool", tool=s["tool"], args=", ".join(str(v)[:60] for v in s["args"].values()), ok=bool(s["ok"]),
                     note="" if s["ok"] else s["result"][:200])
        p, closed = d["proposal"], d["closed"]
        t["agent"] = {"closed": closed, "node": p.node if p else None}
        if closed:
            self.say("closure", ending=closed["kind"], table=closed["source_table"], reason=closed["reason"], drafted_by=self.draft)
            self.state, t["status"] = "idle", "your call"
            return
        said = sql_demo.rehearse(self.wh, p) if p else ""
        if p is None or not said.startswith("verified"):
            self.say("agent", text=("I could not produce a fix that verified in rehearsal"
                                    + (f" ({d['answer'][:200]})" if d["answer"] else "") + ". Nothing was changed."))
            self._needs_person("the agent found no fix that verified")
            return
        self.proposal = p
        self.say("proposal", plan_id=p.plan_id, node=p.node, reason=p.reason, diff=diff(self.wh.deployed(p.node), p.sql),
                 rehearsal=said, tries=d["tries"], refusals=d["refusals"], drafted_by=self.draft)
        self.state, t["status"] = "awaiting", "waiting for your approval"

    def _apply(self, p: Proposal, t: dict) -> None:
        fixer = OpenAICompatModel(**sql_demo._model_settings()) if self.fixer == "model" else None
        runs, case = str(self.dir / "runs.db"), p.case_id
        try:
            if self.crash:
                def hook(event, idx, call, fired={"x": False}):
                    if not fired["x"] and event == "after_tool" and call.tool == "patch_procedure":
                        fired["x"] = True
                        raise SimulatedCrash()
                try:
                    with RunStore(runs) as store:
                        sql_fix.apply(self.wh, store, p, p.plan_id, model=fixer, fault_hook=hook)
                except SimulatedCrash:
                    self.say("step", text="Process killed right after the patch. Starting again from the run log ...", killed=True)
                self.wh = SqlWarehouse(self.dir / "warehouse.duckdb")
            with RunStore(runs) as store:
                run = sql_fix.apply(self.wh, store, p, p.plan_id, model=fixer)
                for line in run["steps"]:
                    self.say("step", text=line)
                out = sql_fix.verify_or_roll_back(self.wh, store, p)
        except NeedsReview as e:
            self.say("agent", text=f"I stopped for a person: {e}. I will not repeat a step that may already have happened.")
            self._needs_person("the apply stopped at a step that is not safe to repeat")
        except Incomplete as e:
            for line in e.steps:
                self.say("step", text=line)
            self.say("agent", text="The apply did not complete. Nothing outside your approval ran, but the plan is only partly carried out.")
            self._needs_person("the approved plan was only partly carried out")
        else:
            self.say("result", kept=bool(out["kept"]), functional=bool(out["functional"]), integrity=bool(out["integrity"]),
                     changes=self.wh.counts()["changes"].get(case, 0), log=self.wh.changes()[-1:] if out["kept"] else [])
            if out["kept"]:
                self._settle(t, f"fixed: {p.node} patched under plan {p.plan_id}")
            else:
                self._needs_person("the fix did not verify and was rolled back")

    def _needs_person(self, why: str) -> None:
        if self.active:
            self.active["status"], self.active["outcome"] = "needs a person", why
        self.state, self.proposal = "idle", None

    def _settle(self, t: dict, outcome: str, by_person: bool = False) -> None:
        """Nothing is closed on the agent's word. The warehouse is compared with its reference, and a
        ticket whose ending left anything wrong stays open."""
        with self.wh.open() as raw:
            wrong = raw.affected()
        with self.lock:
            self.reports = self._read_reports()
        self.state, self.proposal = "idle", None
        if wrong:
            self.say("verdict", ok=False, text=f"Checked against the reference: {len(wrong)} tables are still wrong. {t['id']} stays open for a person.")
            t["status"], t["outcome"] = "needs a person", f"{outcome}, but the warehouse is still wrong"
            return
        case = t["case"]
        ended = ("sql" if (t["agent"] or {}).get("node") else ((t["agent"] or {}).get("closed") or {}).get("kind"))
        right = not by_person and {"sql": "sql", "source_data": "source", "nothing_wrong": "none"}.get(ended) == case["kind"]
        if right and case["kind"] == "source":
            right = t["agent"]["closed"]["source_table"] == case["table"]
        t.update(status="closed", outcome=outcome, right=right, was=WAS[case["kind"]].format(**{"node": "", "table": "", **case}))
        self.say("verdict", ok=True, right=right,
                 text=f"Checked against the reference: every table is right. {t['id']} is closed. It was {t['was']}.")

    def tally(self) -> dict:
        done = [t for t in self.tickets if t["status"] == "closed"]
        return {"night": self.night, "closed": len(done), "by_agent": sum(t["right"] for t in done)}

    def mode(self) -> str:
        if self.draft != "model" and self.fixer != "model":
            return "scripted, no model"
        s = sql_demo._model_settings()
        return f"{s['model']} via {urlparse(s['base_url']).hostname}"


class Complaint(BaseModel):
    text: str
    report: str


class Decision(BaseModel):
    plan_id: str
    approve: bool


class Pick(BaseModel):
    ticket: str
    action: str = ""


def build_app(session: Session) -> FastAPI:
    app = FastAPI(title="Keel analyst chat", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return PAGE.read_text(encoding="utf-8")

    @app.get("/api/meta")
    def meta() -> dict:
        return {"reports": list(REPORTS), "mode": session.mode(), "crash": session.crash, "history": session.history}

    @app.get("/api/report/{name}")
    def report(name: str) -> dict:
        if name not in session.reports:
            raise HTTPException(404, f"no report called {name}")
        return session.reports[name]

    @app.get("/api/events")
    def events(since: int = 0) -> dict:
        return {"state": session.state, "events": session.events[since:], "tickets": session.view(), "tally": session.tally()}

    @app.post("/api/night")
    def night() -> dict:
        t = session.next_night()
        return {"state": session.state, "ticket": t["id"] if t else None}

    @app.post("/api/work")
    def work(p: Pick) -> dict:
        session.work(p.ticket)
        return {"state": session.state}

    @app.post("/api/complain")
    def complain(c: Complaint) -> dict:
        if not c.text.strip():
            raise HTTPException(422, "say what looks wrong")
        session.complain(c.text.strip()[:500], c.report)
        return {"state": session.state}

    @app.post("/api/decide")
    def decide(d: Decision) -> dict:
        session.decide(d.plan_id, d.approve)
        return {"state": session.state}

    @app.post("/api/resolve")
    def resolve(p: Pick) -> dict:
        session.resolve(p.ticket, p.action)
        return {"state": session.state}

    @app.post("/api/reset")
    def reset() -> dict:
        if session.state in ("working", "applying"):
            raise HTTPException(409, "wait for the current step to finish")
        session.reset()
        return {"state": session.state}

    return app


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--break", dest="break_id", help="plant this hand-written break on the first night")
    ap.add_argument("--draft", choices=("model", "reference"), default="model")
    ap.add_argument("--fixer", choices=("model", "replay"), default="model")
    ap.add_argument("--crash", action="store_true", help="kill each apply right after the patch, then resume it")
    ap.add_argument("--history", action="store_true", help="let the agent read what was deployed since the last good run")
    ap.add_argument("--seed", type=int, default=0, help="which sequence of nights to play")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--workdir")
    a = ap.parse_args(argv)
    if "model" in (a.draft, a.fixer):
        sql_demo._model_settings()          # fail here, with the instructions, not in the middle of a chat
    print("Building the warehouse and the list of cases ...")
    session = Session(a.workdir or tempfile.mkdtemp(prefix="keel-chat-"), a.break_id, a.draft, a.fixer, a.crash,
                      seed=a.seed, history=a.history)
    import uvicorn
    print(f"Open http://127.0.0.1:{a.port}  ({session.mode()})")
    uvicorn.run(build_app(session), host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
