"""One incident, end to end, on the real warehouse: a model drafts the fix, you approve it, Keel applies it.

    python -m triage_bridge.sql_demo                      # a model drafts and applies; you type the approval
    python -m triage_bridge.sql_demo --crash              # kill the apply after the patch, then resume it
    python -m triage_bridge.sql_demo --draft reference --fixer replay --approve yes     # no model, no key

  1. A break is deployed into a fresh warehouse (default: a seven-day total that sums six days, which
     passes every health check).
  2. The drafter investigates with read-only tools, uses the checker, rehearses its SQL and submits it.
     It can change nothing: its tools read, and a rehearsal puts everything back.
  3. You read the briefing: the node, the SQL change as a diff, what the rehearsal said, and a plan id.
     You approve or reject that plan. Nothing has been applied yet.
  4. The fixer applies it through the harness. It may only do what you approved, with the SQL you
     approved; every call is in the run log, so a killed apply resumes without repeating anything.
  5. The warehouse is checked. If the report or anything else is wrong, the fix is rolled back.

Models are read from HARNESS_MODEL / HARNESS_BASE_URL / HARNESS_API_KEY, falling back to the TRIAGE_
variables, and default to GPT-OSS 120B on Groq. The key is only ever read from the environment.
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from triage.warehouse import as_dag
from triage.warehouse.pipeline import BY_ID

from harness import OpenAICompatModel, RunStore, SimulatedCrash

from . import sql_fix
from .sql_fix import Incomplete, NeedsReview, Proposal
from .sql_warehouse import SqlWarehouse

GROQ, MODEL = "https://api.groq.com/openai/v1", "openai/gpt-oss-120b"


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(f"HARNESS_{name}") or os.environ.get(f"TRIAGE_{name}") or default


def _model_settings() -> dict:
    key = _env("API_KEY")
    if not key:
        raise SystemExit("Set HARNESS_API_KEY (or TRIAGE_API_KEY) in this shell to use a model, or run with "
                         "--draft reference --fixer replay to see the flow without one.")
    return {"model": _env("MODEL", MODEL), "base_url": _env("BASE_URL", GROQ), "api_key": key}


def rehearse(wh: SqlWarehouse, p: Proposal) -> str:
    with wh.open() as raw:
        prior = raw.sql(p.node)
        raw.set_sql(p.node, p.sql)
        raw.run_cycle()
        out = raw.outcome(p.symptom_node, allowed=(p.node,))
        raw.set_sql(p.node, prior)
        raw.run_cycle()
    if out["functional"] and out["integrity"]:
        return "verified on a rehearsal copy: the report and every other table came out right"
    return "NOT verified: " + ("the report came back but other tables did not" if out["functional"] else "the report is still wrong")


def draft_run(wh: SqlWarehouse, incident: dict, how: str) -> dict:
    """Draft a fix and say how it went: the proposal (or None), each tool call, and the counts. The
    terminal demo prints this and the chat shows it; neither decides anything the other does not."""
    if how == "reference":
        node = next(n for n in BY_ID if wh.deployed(n) != BY_ID[n].sql)
        p = Proposal(incident["id"], incident["symptom_node"], node, BY_ID[node].sql, "restored the reference definition")
        return {"proposal": p, "steps": [], "tries": 0, "refusals": 0, "answer": ""}
    from triage.investigator import OpenAICompatChat
    from triage.warehouse import fix
    with wh.open() as raw:
        inv, sql, session = fix.draft_checked(raw, as_dag(), incident, OpenAICompatChat(**_model_settings()))
    p = None if sql is None else Proposal(incident["id"], incident["symptom_node"], inv.fix_target, sql, inv.reason)
    return {"proposal": p, "steps": inv.steps, "tries": session.tries, "refusals": len(session.refusals), "answer": inv.answer}


def draft(wh: SqlWarehouse, incident: dict, how: str, break_id: str) -> Proposal | None:
    d = draft_run(wh, incident, how)
    if how == "reference":
        return d["proposal"]
    for s in d["steps"]:
        print(f"    {s['tool']}({', '.join(str(v)[:40] for v in s['args'].values())}) {'ok' if s['ok'] else 'REFUSED: ' + s['result'][:90]}")
    print(f"    rehearsal tries: {d['tries']}; refused by the harness: {d['refusals']}")
    if d["proposal"] is None:
        print(f"\nThe drafter did not produce a verified fix ({d['answer'][:160] or 'no answer'}). This goes to a person.")
    return d["proposal"]


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--break", dest="break_id", default="date_window.trailing_six_days")
    ap.add_argument("--draft", choices=("model", "reference"), default="model")
    ap.add_argument("--fixer", choices=("model", "replay"), default="model")
    ap.add_argument("--approve", choices=("ask", "yes", "no"), default="ask")
    ap.add_argument("--crash", action="store_true", help="kill the apply right after the patch, then resume it")
    ap.add_argument("--workdir")
    a = ap.parse_args(argv)
    work = Path(a.workdir or tempfile.mkdtemp(prefix="keel-sql-"))
    work.mkdir(parents=True, exist_ok=True)
    report: dict = {"break": a.break_id, "workdir": str(work)}

    print(f"1. Deploying the break {a.break_id} into a fresh warehouse ...")
    wh = SqlWarehouse.create(work / "warehouse.duckdb", a.break_id)
    incident = wh.incident("CASE-1")
    print(f"   Complaint: {incident['symptom']}")
    flagged = {n: s["status"] for n, s in incident["node_status"].items() if s["status"] != "ok"}
    print(f"   Health checks: {flagged or 'every node passes'}")

    print(f"\n2. Drafting a fix ({a.draft}) ...")
    p = draft(wh, incident, a.draft, a.break_id)
    if p is None:
        return {**report, "result": "no verified fix"}
    said = rehearse(wh, p)

    print("\n3. For your approval. Nothing has been applied.\n")
    print(sql_fix.briefing(wh, p, said) + "\n")
    answer = a.approve if a.approve != "ask" else input(f"Approve plan {p.plan_id}? [y/N] ").strip().lower()
    approved = answer in ("y", "yes")
    report.update({"plan_id": p.plan_id, "node": p.node, "approved": approved})
    if not approved:
        print("Rejected. Nothing was applied.")
        return {**report, "result": "rejected", "deployed_unchanged": wh.deployed(p.node) != p.sql}

    print(f"\n4. Applying plan {p.plan_id} through the harness ({a.fixer}) ...")
    fixer = OpenAICompatModel(**_model_settings()) if a.fixer == "model" else None
    runs = str(work / "runs.db")
    try:
        if a.crash:
            def hook(event, idx, call, fired={"x": False}):
                if not fired["x"] and event == "after_tool" and call.tool == "patch_procedure":
                    fired["x"] = True
                    raise SimulatedCrash()
            try:
                with RunStore(runs) as store:
                    sql_fix.apply(wh, store, p, p.plan_id, model=fixer, fault_hook=hook)
            except SimulatedCrash:
                print("   Process killed right after the patch. Starting again from the run log ...")
            wh = SqlWarehouse(work / "warehouse.duckdb")
        with RunStore(runs) as store:
            run = sql_fix.apply(wh, store, p, p.plan_id, model=fixer)
            for line in run["steps"]:
                print(f"   {line}")
            print("\n5. Checking the warehouse ...")
            out = sql_fix.verify_or_roll_back(wh, store, p)
    except NeedsReview as e:
        print(f"   Stopped for a person: {e}")
        return {**report, "result": "needs review"}
    except Incomplete as e:
        for line in e.steps:
            print(f"   {line}")
        state = wh.outcome(p.symptom_node, allowed=(p.node,))
        print(f"\n   The apply did NOT complete ({e.answer}).")
        print(f"   Nothing outside the approval ran. As it stands: report right {state['functional']}, "
              f"everything else right {state['integrity']}, change log lines {wh.counts()['changes'].get('CASE-1', 0)}.")
        print("   This goes to a person: the approved plan is only partly carried out.")
        return {**report, "result": "incomplete", "status": e.status}
    print(f"   Report right: {out['functional']}. Everything else right: {out['integrity']}.")
    print("   Kept." if out["kept"] else "   Rolled back to the prior definition.")
    counts = wh.counts()
    print(f"   Change log lines for this case: {counts['changes'].get('CASE-1', 0)}; text: {wh.changes()}")
    return {**report, "result": "kept" if out["kept"] else "rolled back", "status": run["status"],
            "functional": out["functional"], "integrity": out["integrity"], "changes": counts["changes"].get("CASE-1", 0)}


if __name__ == "__main__":
    main()
