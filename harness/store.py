"""Durable run log in SQLite. One row per step, written before and after every tool call.

The two writes are the whole crash-recovery story:

* 'started'     intent recorded, tool not yet called
* 'dispatched'  written immediately before the tool is invoked
* 'done'        written after, with the result

On resume, 'done' steps are replayed from the log (their results are reused, the tool is
not called again). A step left at 'started' never ran, so it is safe to run. A step left at
'dispatched' may or may not have taken effect: the harness repeats it only if the tool is
idempotent, and otherwise stops the run for a human to check.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = """
create table if not exists runs (
  id text primary key, agent text not null, task text not null, task_kind text,
  status text not null, answer text, created real not null, updated real not null
);
create table if not exists steps (
  run_id text not null, idx integer not null, tool text, args text, idempotent integer,
  status text not null, result text, error text, attempts integer default 0,
  started real, finished real, primary key (run_id, idx)
);
"""


class RunStore:
    def __init__(self, path: str | Path = ":memory:"):
        self.db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        """Release the file. Windows will not let an open file be deleted, so a store that
        outlives its directory is a portability bug rather than an untidy detail."""
        self.db.close()

    def __enter__(self) -> RunStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # runs
    def create_run(self, agent: str, task: str, task_kind: str | None = None, run_id: str | None = None) -> str:
        rid = run_id or f"run-{uuid.uuid4().hex[:10]}"
        now = time.time()
        self.db.execute("insert into runs values (?,?,?,?,?,?,?,?)", (rid, agent, task, task_kind, "running", None, now, now))
        return rid

    def get_run(self, run_id: str) -> dict | None:
        r = self.db.execute("select * from runs where id=?", (run_id,)).fetchone()
        return dict(r) if r else None

    def set_status(self, run_id: str, status: str, answer: str | None = None) -> None:
        self.db.execute("update runs set status=?, answer=coalesce(?, answer), updated=? where id=?",
                        (status, answer, time.time(), run_id))

    def runs(self, status: str | None = None) -> list[dict]:
        q, p = "select * from runs", ()
        if status:
            q, p = q + " where status=?", (status,)
        return [dict(r) for r in self.db.execute(q + " order by created", p)]

    # steps
    def begin_step(self, run_id: str, idx: int, tool: str, args: dict, idempotent: bool) -> None:
        self.db.execute(
            "insert or replace into steps (run_id, idx, tool, args, idempotent, status, started, attempts) "
            "values (?,?,?,?,?, 'started', ?, coalesce((select attempts from steps where run_id=? and idx=?), 0))",
            (run_id, idx, tool, json.dumps(args, sort_keys=True), int(idempotent), time.time(), run_id, idx))

    def bump_attempts(self, run_id: str, idx: int) -> None:
        self.db.execute("update steps set attempts = attempts + 1 where run_id=? and idx=?", (run_id, idx))

    def mark_dispatched(self, run_id: str, idx: int) -> None:
        self.db.execute("update steps set status='dispatched' where run_id=? and idx=?", (run_id, idx))

    def finish_step(self, run_id: str, idx: int, result: Any = None, error: str | None = None) -> None:
        self.db.execute("update steps set status=?, result=?, error=?, finished=? where run_id=? and idx=?",
                        ("error" if error else "done", json.dumps(result, default=str), error, time.time(), run_id, idx))

    def resolve_review(self, run_id: str, idx: int, took_effect: bool, note: str = "") -> None:
        """A person checked a step the harness refused to guess about.

        took_effect=True records it as done, so resume moves past it. False puts it back to
        'started' (it never ran), so resume runs it once.
        """
        if took_effect:
            self.db.execute("update steps set status='done', result=?, finished=? where run_id=? and idx=?",
                            (json.dumps(note or "confirmed by a person"), time.time(), run_id, idx))
        else:
            self.db.execute("update steps set status='started' where run_id=? and idx=?", (run_id, idx))
        self.db.execute("update runs set status='running', updated=? where id=?", (time.time(), run_id))

    def steps(self, run_id: str) -> list[dict]:
        rows = self.db.execute("select * from steps where run_id=? order by idx", (run_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["args"] = json.loads(d["args"]) if d["args"] else {}
            d["result"] = json.loads(d["result"]) if d["result"] else None
            out.append(d)
        return out
