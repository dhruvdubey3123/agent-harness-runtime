"""A small stand-in for the systems a fix touches: the warehouse and the ticketing tool.

It is a SQLite file, so its state survives a process kill exactly like a real system would.
Each action is either safe to repeat (patching a procedure to a known version, rebuilding a
table) or not (re-running a load appends rows, a ticket comment is posted again).
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel

from harness.tools import ToolRegistry

SCHEMA = """
create table if not exists procedures (node text primary key, patched_for text, patched_at real);
create table if not exists job_runs   (id integer primary key autoincrement, node text, at real);
create table if not exists refreshes  (node text primary key, at real);
create table if not exists comments   (id integer primary key autoincrement, case_id text, body text, at real);
create table if not exists audit      (id integer primary key autoincrement, line text, at real);
"""


class Node(BaseModel):
    node: str


class Patch(BaseModel):
    node: str
    change: str | None = None     # under an approval, supplied by the runtime from what a person approved


class Comment(BaseModel):
    case_id: str
    body: str


class Audit(BaseModel):
    line: str


class Warehouse:
    def __init__(self, path: str | Path):
        self.path = str(path)
        with self._db() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def _db(self):
        """One connection per call, because tools run in worker threads, and closed on the way out:
        `with sqlite3.connect(...)` commits but does not close, and an open file cannot be deleted
        on Windows."""
        conn = sqlite3.connect(self.path, isolation_level=None)
        try:
            yield conn
        finally:
            conn.close()

    # ---------------------------------------------------------------- what an auditor would check
    def counts(self) -> dict:
        with self._db() as db:
            q = lambda sql: db.execute(sql).fetchall()
            return {
                "job_runs": dict(q("select node, count(*) from job_runs group by node")),
                "comments": dict(q("select case_id, count(*) from comments group by case_id")),
                "audit_lines": dict(q("select line, count(*) from audit group by line")),
                "patched": [r[0] for r in q("select node from procedures")],
                "refreshed": [r[0] for r in q("select node from refreshes")],
            }

    def texts(self) -> dict:
        """What was written into the warehouse. Comments and audit lines are the fixer's own wording;
        a patch is not: its content is whatever a person approved."""
        with self._db() as db:
            return {"comments": [r[0] for r in db.execute("select body from comments")],
                    "audit": [r[0] for r in db.execute("select line from audit")],
                    "patches": {r[0]: r[1] for r in db.execute("select node, patched_for from procedures")}}

    def duplicated_actions(self) -> int:
        """Side effects that happened more than once: extra load runs, extra comments, extra audit lines."""
        c = self.counts()
        return (sum(n - 1 for n in c["job_runs"].values() if n > 1)
                + sum(n - 1 for n in c["comments"].values() if n > 1)
                + sum(n - 1 for n in c["audit_lines"].values() if n > 1))

    def took_effect(self, tool: str, args: dict) -> bool:
        """What a person checks before telling the harness a step went through."""
        with self._db() as db:
            if tool == "rerun_job":
                return db.execute("select count(*) from job_runs where node=?", (args["node"],)).fetchone()[0] > 0
            if tool == "comment_ticket":
                return db.execute("select count(*) from comments where case_id=?", (args["case_id"],)).fetchone()[0] > 0
            if tool == "append_audit":
                return db.execute("select count(*) from audit where line=?", (args["line"],)).fetchone()[0] > 0
        return False

    # ---------------------------------------------------------------- tools
    def tools(self) -> ToolRegistry:
        reg = ToolRegistry()

        @reg.tool("patch_procedure", "Deploy the reviewed fix to a stored procedure (CREATE OR REPLACE). Safe to repeat.",
                  Patch, idempotent=True, side_effects=True)
        def patch_procedure(node: str, change: str | None = None) -> str:
            change = change or "unspecified"
            with self._db() as db:
                db.execute("insert into procedures values (?,?,?) on conflict(node) do update set patched_for=excluded.patched_for, "
                           "patched_at=excluded.patched_at", (node, change, time.time()))
            return f"{node} patched ({change})"

        @reg.tool("rerun_job", "Run a load or procedure once. NOT safe to repeat: a second run appends the rows again.",
                  Node, idempotent=False, side_effects=True)
        def rerun_job(node: str) -> str:
            with self._db() as db:
                db.execute("insert into job_runs (node, at) values (?,?)", (node, time.time()))
            return f"{node} ran"

        @reg.tool("refresh_table", "Rebuild a table or report from its inputs. Safe to repeat.", Node,
                  idempotent=True, side_effects=True)
        def refresh_table(node: str) -> str:
            with self._db() as db:
                db.execute("insert into refreshes values (?,?) on conflict(node) do update set at=excluded.at", (node, time.time()))
            return f"{node} refreshed"

        @reg.tool("comment_ticket", "Post the resolution on the incident ticket. NOT safe to repeat.", Comment,
                  idempotent=False, side_effects=True)
        def comment_ticket(case_id: str, body: str) -> str:
            with self._db() as db:
                db.execute("insert into comments (case_id, body, at) values (?,?,?)", (case_id, body, time.time()))
            return f"commented on {case_id}"

        @reg.tool("append_audit", "Write one line to the change audit log. NOT safe to repeat.", Audit,
                  idempotent=False, side_effects=True)
        def append_audit(line: str) -> str:
            with self._db() as db:
                db.execute("insert into audit (line, at) values (?,?)", (line, time.time()))
            return "audit line written"

        return reg
