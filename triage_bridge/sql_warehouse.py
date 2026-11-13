"""The real warehouse behind Keel: DuckDB, twelve SQL procedures, and a fix that is SQL.

triage_bridge/warehouse.py is a stand-in: a patch there is a sentence written into a row. This is
the warehouse from pipeline-triage-graph (TPC-H in DuckDB, real procedure SQL, real breaks), with
three tools a fix needs:

    patch_procedure   replace one procedure's SQL. Safe to repeat: deploying the same definition
                      twice changes nothing. The definition it replaces is written to a history
                      table first, so a rollback is a replay of what was there.
    run_cycle         rebuild every table from its procedure, in dependency order. Safe to repeat.
    record_change     append a line to the change log. Not safe to repeat: a second call is a
                      second line, which is the kind of action a crash must not duplicate.

The warehouse is a file and every call opens it, acts and closes it, so what a killed process leaves
behind is exactly what the next process finds.
"""
from __future__ import annotations

import re
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel
from triage.warehouse import BREAKS, Warehouse, observe
from triage.warehouse.pipeline import BY_ID, REPORTS

from harness.tools import NoArgs, ToolRegistry
from harness.types import ToolCall

EXTRA = """
create table if not exists procedure_history (seq integer, node text, prior_sql text, new_sql text);
create table if not exists change_log (seq integer, case_id text, body text);
"""


class Patch(BaseModel):
    node: str
    sql: str | None = None      # under an approval, supplied by the runtime from what a person approved


class Change(BaseModel):
    case_id: str
    body: str


class SqlWarehouse:
    def __init__(self, path: str | Path):
        self.path = str(path)

    @classmethod
    def create(cls, path: str | Path, break_id: str | None = None) -> "SqlWarehouse":
        """Build the reference warehouse, and optionally deploy one of the known breaks into it."""
        wh = Warehouse(str(path)).build()
        wh.db.execute(EXTRA)
        if break_id:
            b = next(x for x in BREAKS if x.id == break_id)
            wh.set_sql(b.node, b.broken_sql())
            wh.run_cycle()
        wh.close()
        return cls(path)

    @contextmanager
    def open(self):
        wh = Warehouse(self.path)
        wh.cycle = wh.db.execute("select coalesce(max(cycle), 0) from run_log").fetchone()[0]
        try:
            yield wh
        finally:
            wh.close()

    # ---------------------------------------------------------------- what a person, or a test, looks at

    def deployed(self, node: str) -> str:
        with self.open() as wh:
            return wh.sql(node)

    def incident(self, case_id: str) -> dict | None:
        """The complaint as a person would file it, or None when every report is right."""
        with self.open() as wh:
            wrong = wh.affected()
            symptom = next((r for r in REPORTS if r in wrong), None)
            if symptom is None:
                return None
            return {"id": case_id, "symptom_node": symptom, "symptom": wh.complaint(symptom), "node_status": observe(wh)}

    def outcome(self, symptom_node: str, allowed: tuple[str, ...]) -> dict:
        with self.open() as wh:
            return wh.outcome(symptom_node, allowed=allowed)

    def counts(self) -> dict:
        with self.open() as wh:
            q = lambda sql: wh.db.execute(sql).fetchall()
            return {"cycles": wh.cycle,
                    "changes": dict(q("select case_id, count(*) from change_log group by case_id")),
                    "patches": [r[0] for r in q("select node from procedure_history order by seq")]}

    def changes(self) -> list[str]:
        with self.open() as wh:
            return [r[0] for r in wh.db.execute("select body from change_log order by seq").fetchall()]

    def took_effect(self, tool: str, args: dict) -> bool:
        """What a person checks before telling the harness an interrupted step went through."""
        if tool != "record_change":
            return False
        with self.open() as wh:
            return wh.db.execute("select count(*) from change_log where case_id = ? and body = ?",
                                 [args["case_id"], args["body"]]).fetchone()[0] > 0

    def rollback_calls(self, case_id: str, node: str) -> list[ToolCall]:
        """The calls that put back what the last patch of `node` replaced."""
        with self.open() as wh:
            row = wh.db.execute("select prior_sql from procedure_history where node = ? order by seq desc limit 1",
                                [node]).fetchone()
        if row is None:
            raise LookupError(f"{node} was never patched here, so there is nothing to roll back")
        return [ToolCall("patch_procedure", {"node": node, "sql": row[0]}), ToolCall("run_cycle", {}),
                ToolCall("record_change", {"case_id": case_id, "body": f"rolled back the change to {node}: it did not verify"})]

    # ---------------------------------------------------------------- tools

    def tools(self) -> ToolRegistry:
        reg = ToolRegistry()

        @reg.tool("patch_procedure", "Deploy the approved SQL to one procedure. Safe to repeat. Leave `sql` out: the "
                  "approved SQL is supplied for you.", Patch, idempotent=True, side_effects=True)
        def patch_procedure(node: str, sql: str | None = None) -> str:
            if node not in BY_ID:
                raise ValueError(f"'{node}' is not a procedure in this warehouse")
            if not sql or ";" in sql.strip().rstrip(";") or not re.match(r"(?is)^\s*(select|with)\b", sql):
                raise ValueError("a procedure is one SELECT statement")
            sql = sql.strip().rstrip(";").strip()
            with self.open() as wh:
                prior = wh.sql(node)
                if prior == sql:
                    return f"{node} already has this definition"
                seq = wh.db.execute("select coalesce(max(seq), 0) + 1 from procedure_history").fetchone()[0]
                wh.db.execute("insert into procedure_history values (?, ?, ?, ?)", [seq, node, prior, sql])
                wh.set_sql(node, sql)
            return f"{node} patched; the definition it replaced is kept in the history"

        @reg.tool("run_cycle", "Rebuild every table from its procedure, in dependency order. Safe to repeat.", NoArgs,
                  idempotent=True, side_effects=True)
        def run_cycle() -> str:
            with self.open() as wh:
                result = wh.run_cycle()
            bad = {n: s for n, s in result.items() if s != "success"}
            return f"{len(result) - len(bad)} of {len(result)} procedures succeeded" + (f"; not run: {bad}" if bad else "")

        @reg.tool("record_change", "Append one line to the change log saying what was done. NOT safe to repeat.",
                  Change, idempotent=False, side_effects=True)
        def record_change(case_id: str, body: str) -> str:
            with self.open() as wh:
                seq = wh.db.execute("select coalesce(max(seq), 0) + 1 from change_log").fetchone()[0]
                wh.db.execute("insert into change_log values (?, ?, ?)", [seq, case_id, body])
            return f"change recorded for {case_id}"

        return reg
