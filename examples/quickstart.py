"""Run an agent, kill it halfway, and resume it from the log.

    python examples/quickstart.py
"""
import tempfile
from pathlib import Path

from harness import AgentSpec, Harness, Policy, RunStore, ScriptedModel, SimulatedCrash, ToolCall, workspace_tools

script = [
    ToolCall("write_file", {"path": "notes.md", "content": "# Notes"}),
    ToolCall("append_line", {"path": "audit.log", "line": "notes created"}),
    ToolCall("read_file", {"path": "notes.md"}),
    ToolCall("append_line", {"path": "audit.log", "line": "notes checked"}),
]
agent = AgentSpec("writer", "Keep notes.", Policy({"write_file", "append_line", "read_file"}))

d = Path(tempfile.mkdtemp())
db = d / "runs.db"


def crash_on_step_2(event, idx, call):
    if event == "before_tool" and idx == 2:
        raise SimulatedCrash()


try:
    Harness(workspace_tools(d / "ws"), ScriptedModel(script), RunStore(db), fault_hook=crash_on_step_2).start(agent, "keep notes", run_id="demo")
except SimulatedCrash:
    print("crashed at step 2")

h = Harness(workspace_tools(d / "ws"), ScriptedModel(script), RunStore(db))
h.resume(agent, "demo")
for s in h.store.steps("demo"):
    print(s["idx"], s["tool"], s["status"])
print("status:", h.store.get_run("demo")["status"], "| replayed from log:", h.metrics["demo"].replayed_from_log)
print("audit.log:", (d / "ws/audit.log").read_text().splitlines())
