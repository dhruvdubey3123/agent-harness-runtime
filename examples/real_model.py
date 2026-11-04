"""Drive the harness with a real open-weight model and check the input/output end to end.

Works with any OpenAI-compatible server. With Ollama:

    ollama pull qwen3:8b
    pip install -e ".[local]"
    python examples/real_model.py                      # HARNESS_MODEL / HARNESS_BASE_URL to override

Four checks, each on a fresh workspace:
  1. clean run          the model reads an incident, writes a fix plan, appends one audit line
  2. crash, safe step   killed right after write_file ran (idempotent): resume repeats it and finishes
  3. crash, risky step  killed right after append_line ran (not idempotent): resume stops for review,
                        while restarting from scratch writes the audit line twice
  4. sandbox            a read-only agent asked to write gets refused, and the refusal is logged
A JSON report is written to examples/real_model_report.json.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import AgentSpec, Harness, OpenAICompatModel, Policy, RunStore, SimulatedCrash, workspace_tools  # noqa: E402

INCIDENT = """# Case AGS-4471: revenue dashboard is empty this morning

Failed job: sp_load_daily_revenue
Error: SQL compilation error: invalid identifier 'CUST_ID'
Note: yesterday the upstream table CRM.CUSTOMERS renamed column CUST_ID to CUSTOMER_ID.
"""
TASK = ("Read incident.md. Write fix_plan.md containing the root cause and the one-line SQL fix. "
        "Then append exactly one line to audit.log saying the fix plan was written. "
        "Then finish with a one-sentence summary. Use each tool only once.")

WRITER = AgentSpec("writer", "You are a careful data-pipeline triage agent. Use the tools to do the task, one call at a time.",
                   Policy({"read_file", "list_files", "write_file", "append_line"}), max_steps=8)
READER = AgentSpec("reader", "You may only read files.", Policy({"read_file", "list_files"}), max_steps=4)

model = OpenAICompatModel()
report: dict = {"model": model.model, "base_url": model.base_url, "checks": {}}


def fresh() -> tuple[Path, RunStore]:
    d = Path(tempfile.mkdtemp(prefix="harness_real_"))
    (d / "ws").mkdir()
    (d / "ws" / "incident.md").write_text(INCIDENT)
    return d, RunStore(str(d / "runs.db"))


def crash_after(tool_name: str):
    fired = {"done": False}

    def hook(event, idx, call):
        if event == "after_tool" and call.tool == tool_name and not fired["done"]:
            fired["done"] = True
            raise SimulatedCrash()   # the tool already ran; the log has not recorded it as done
    return hook


def audit_lines(d: Path) -> list[str]:
    p = d / "ws" / "audit.log"
    return p.read_text().splitlines() if p.exists() else []


def summary(store: RunStore, rid: str) -> dict:
    run = store.get_run(rid)
    return {"status": run["status"], "answer": run.get("answer") or run.get("detail"),
            "steps": [f'{s["idx"]} {s["tool"]} -> {s["status"]}' for s in store.steps(rid)]}


def show(title: str, info: dict) -> None:
    print(f"\n=== {title}")
    for k, v in info.items():
        if isinstance(v, list):
            print(f"  {k}:")
            for x in v:
                print(f"    {x}")
        else:
            print(f"  {k}: {v}")


t0 = time.time()

# 1. clean run
d, store = fresh()
h = Harness(workspace_tools(d / "ws"), model, store)
rid = h.start(WRITER, TASK, run_id="clean")
info = summary(store, rid)
plan = d / "ws" / "fix_plan.md"
info["fix_plan.md"] = plan.read_text().strip()[:400] if plan.exists() else "(missing)"
info["audit.log"] = audit_lines(d)
info["pass"] = info["status"] == "done" and plan.exists() and len(info["audit.log"]) == 1
report["checks"]["clean_run"] = info
show("1. clean run", info)

# 2. crash after the idempotent write
d, store = fresh()
try:
    Harness(workspace_tools(d / "ws"), model, store, fault_hook=crash_after("write_file")).start(WRITER, TASK, run_id="r2")
    crashed = False
except SimulatedCrash:
    crashed = True
h = Harness(workspace_tools(d / "ws"), model, store)
h.resume(WRITER, "r2")
info = {"crashed": crashed, **summary(store, "r2"), "audit.log": audit_lines(d),
        "replayed_from_log": h.metrics["r2"].replayed_from_log}
info["pass"] = crashed and info["status"] == "done" and len(info["audit.log"]) == 1
report["checks"]["crash_after_idempotent_write"] = info
show("2. killed right after write_file, then resumed", info)

# 3. crash after the non-idempotent append: resume vs restart
d, store = fresh()
try:
    Harness(workspace_tools(d / "ws"), model, store, fault_hook=crash_after("append_line")).start(WRITER, TASK, run_id="r3")
    crashed = False
except SimulatedCrash:
    crashed = True
lines_at_crash = audit_lines(d)
h = Harness(workspace_tools(d / "ws"), model, store)
h.resume(WRITER, "r3")
resumed = {"crashed": crashed, **summary(store, "r3"), "audit.log": audit_lines(d)}
Harness(workspace_tools(d / "ws"), model, RunStore(str(d / "restart.db"))).start(WRITER, TASK, run_id="restart")
restarted_lines = audit_lines(d)
info = {"audit.log at crash": lines_at_crash, "resume": resumed["status"], "resume steps": resumed["steps"],
        "audit.log after resume": resumed["audit.log"], "audit.log after restarting from scratch": restarted_lines}
info["pass"] = crashed and resumed["status"] == "needs_review" and len(resumed["audit.log"]) == 1
report["checks"]["crash_after_side_effect"] = info
show("3. killed right after append_line: resume vs restart", info)

# 4. sandbox
d, store = fresh()
Harness(workspace_tools(d / "ws"), model, store).start(READER, "Write the word hello to notes.md.", run_id="r4")
steps = store.steps("r4")
info = {**summary(store, "r4"), "files": sorted(p.name for p in (d / "ws").iterdir()),
        "errors_shown_to_model": [s.get("error") for s in steps if s.get("error")]}
info["pass"] = not (d / "ws" / "notes.md").exists()
report["checks"]["sandbox_read_only_agent"] = info
show("4. read-only agent asked to write", info)

report["seconds"] = round(time.time() - t0, 1)
report["passed"] = sum(c["pass"] for c in report["checks"].values())
out = Path(__file__).with_name("real_model_report.json")
out.write_text(json.dumps(report, indent=2, default=str))
print(f"\n{report['passed']}/4 checks passed with {model.model} in {report['seconds']} s. Report: {out}")
