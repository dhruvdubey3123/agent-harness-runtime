"""The analyst's side of it: a dashboard with a number that looks wrong, and a chat beside it.

    python -m triage_bridge.chat                                   # a model drafts and applies
    python -m triage_bridge.chat --draft reference --fixer replay   # no model, no key
    python -m triage_bridge.chat --crash                            # kill the apply after the patch, then resume

Then open http://127.0.0.1:8765. Say what looks wrong with the report on screen. The agent
investigates with read-only tools, rehearses a fix, and puts it to you in the chat with a diff. Nothing
is applied until you approve that plan; then Keel applies it through its run log, the warehouse is
checked, and the dashboard reloads.

This is the same flow as triage_bridge.sql_demo with a different front door. It does one thing: a
complaint about a report, to a fix a person approves. It is not a general assistant for the
warehouse, and it does not answer questions it has no tool for.

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

from triage.warehouse.pipeline import REPORTS

from harness import OpenAICompatModel, RunStore, SimulatedCrash

from . import sql_demo, sql_fix
from .sql_fix import Incomplete, NeedsReview, Proposal
from .sql_warehouse import SqlWarehouse

PAGE = Path(__file__).parent / "data" / "chat.html"
BREAKS_AT = re.compile(r"\s+(?=(?:from|where|group by|order by|left join|inner join|join|having|union)\b)", re.I)


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
    """One warehouse, one conversation. State moves idle -> working -> awaiting -> applying -> idle.
    Everything the page shows is an event in `events`; the page polls for the ones it has not seen."""

    def __init__(self, workdir: str | Path, break_id: str, draft: str = "model", fixer: str = "model",
                 crash: bool = False, background: bool = True):
        self.work = Path(workdir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.break_id, self.draft, self.fixer, self.crash, self.background = break_id, draft, fixer, crash, background
        self.lock = threading.Lock()
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        with self.lock:
            for f in self.work.glob("*.d*b*"):
                f.unlink()
            self.wh = SqlWarehouse.create(self.work / "warehouse.duckdb", self.break_id)
            self.events: list[dict] = []
            self.state, self.proposal, self.case = "idle", None, 0
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
        self.events.append({"n": len(self.events), "kind": kind, **body})

    def _go(self, fn, *args) -> None:
        def guarded():
            try:
                fn(*args)
            except BaseException as e:      # a model with no key raises SystemExit; the page must still hear about it
                self.say("agent", text=f"I had to stop: {e}. Nothing was applied beyond what the log above shows.")
                self.state = "idle"
        if self.background:
            threading.Thread(target=guarded, daemon=True).start()
        else:
            guarded()

    # ------------------------------------------------------------------ what a person can do
    def complain(self, text: str, report: str) -> None:
        if self.state != "idle":
            raise HTTPException(409, f"I'm {self.state.replace('awaiting', 'waiting for your decision')}; one case at a time.")
        if report not in REPORTS:
            raise HTTPException(404, f"no report called {report}")
        self.state, self.case = "working", self.case + 1
        self.say("user", text=text)
        self._go(self._investigate, text, report)

    def decide(self, plan_id: str, approve: bool) -> None:
        p = self.proposal
        if self.state != "awaiting" or p is None:
            raise HTTPException(409, "there is no plan waiting for a decision")
        if plan_id != p.plan_id:
            raise HTTPException(409, f"plan {plan_id} is not the plan on the table ({p.plan_id})")
        if not approve:
            self.say("user", text=f"Reject plan {p.plan_id}")
            self.say("agent", text="Rejected. Nothing was applied, and the warehouse is as it was. This case goes to a person.")
            self.state, self.proposal = "idle", None
            return
        self.state = "applying"
        self.say("user", text=f"Approve plan {p.plan_id}")
        self._go(self._apply, p)

    # ------------------------------------------------------------------ the work
    def _investigate(self, text: str, report: str) -> None:
        incident = self.wh.incident(f"CASE-{self.case}")
        if incident is None:
            self.say("agent", text="Every report matches its reference run, so I have nothing to trace. If a number still looks wrong, it is wrong in the source data, which I can't see.")
            self.state = "idle"
            return
        incident.update({"symptom_node": report, "symptom": f"an analyst looking at {report} says: {text}"})
        self.say("agent", text=f"Looking into {report}. While I investigate I can read tables and their SQL, and I can't change anything.")
        d = sql_demo.draft_run(self.wh, incident, self.draft)
        for s in d["steps"]:
            self.say("tool", tool=s["tool"], args=", ".join(str(v)[:60] for v in s["args"].values()), ok=bool(s["ok"]),
                     note="" if s["ok"] else s["result"][:160])
        p = d["proposal"]
        said = sql_demo.rehearse(self.wh, p) if p else ""
        if p is None or not said.startswith("verified"):
            self.say("agent", text=("I could not produce a fix that verified in rehearsal"
                                    + (f" ({d['answer'][:200]})" if d["answer"] else "") + ". Nothing was changed. This goes to a person."))
            self.state = "idle"
            return
        self.proposal = p
        self.say("proposal", plan_id=p.plan_id, node=p.node, reason=p.reason, diff=diff(self.wh.deployed(p.node), p.sql),
                 rehearsal=said, tries=d["tries"], refusals=d["refusals"], drafted_by=self.draft)
        self.state = "awaiting"

    def _apply(self, p: Proposal) -> None:
        fixer = OpenAICompatModel(**sql_demo._model_settings()) if self.fixer == "model" else None
        runs, case = str(self.work / "runs.db"), p.case_id
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
                self.wh = SqlWarehouse(self.work / "warehouse.duckdb")
            with RunStore(runs) as store:
                run = sql_fix.apply(self.wh, store, p, p.plan_id, model=fixer)
                for line in run["steps"]:
                    self.say("step", text=line)
                out = sql_fix.verify_or_roll_back(self.wh, store, p)
        except NeedsReview as e:
            self.say("agent", text=f"I stopped for a person: {e}. I will not repeat a step that may already have happened.")
        except Incomplete as e:
            for line in e.steps:
                self.say("step", text=line)
            self.say("agent", text="The apply did not complete. Nothing outside your approval ran, but the plan is only partly carried out. This goes to a person.")
        else:
            self.say("result", kept=bool(out["kept"]), functional=bool(out["functional"]), integrity=bool(out["integrity"]),
                     changes=self.wh.counts()["changes"].get(case, 0), log=self.wh.changes()[-1:] if out["kept"] else [])
        with self.lock:
            self.reports = self._read_reports()
        self.state, self.proposal = "idle", None

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


def build_app(session: Session) -> FastAPI:
    app = FastAPI(title="Keel analyst chat", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return PAGE.read_text(encoding="utf-8")

    @app.get("/api/meta")
    def meta() -> dict:
        return {"reports": list(REPORTS), "mode": session.mode(), "crash": session.crash}

    @app.get("/api/report/{name}")
    def report(name: str) -> dict:
        if name not in session.reports:
            raise HTTPException(404, f"no report called {name}")
        return session.reports[name]

    @app.get("/api/events")
    def events(since: int = 0) -> dict:
        return {"state": session.state, "events": session.events[since:]}

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

    @app.post("/api/reset")
    def reset() -> dict:
        if session.state in ("working", "applying"):
            raise HTTPException(409, "wait for the current step to finish")
        session.reset()
        return {"state": session.state}

    return app


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--break", dest="break_id", default="type_drift.price_as_integer")
    ap.add_argument("--draft", choices=("model", "reference"), default="model")
    ap.add_argument("--fixer", choices=("model", "replay"), default="model")
    ap.add_argument("--crash", action="store_true", help="kill the apply right after the patch, then resume it")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--workdir")
    a = ap.parse_args(argv)
    if "model" in (a.draft, a.fixer):
        sql_demo._model_settings()          # fail here, with the instructions, not in the middle of a chat
    session = Session(a.workdir or tempfile.mkdtemp(prefix="keel-chat-"), a.break_id, a.draft, a.fixer, a.crash)
    import uvicorn
    print(f"Open http://127.0.0.1:{a.port}  ({session.mode()})")
    uvicorn.run(build_app(session), host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
