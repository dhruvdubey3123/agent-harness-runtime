from pathlib import Path

import pytest

from harness import (AgentSpec, ContextBudget, Finish, Harness, Policy, RunStore, ScriptedModel, SimulatedCrash,
                     ToolCall, distill, load_skills_index, workspace_tools)
from harness.context import build_context
from harness.tools import Tool, ToolRegistry
from harness.types import RetryableToolError

from pydantic import BaseModel

WRITER = Policy({"read_file", "list_files", "write_file", "append_line"})


def agent(policy=WRITER, **kw):
    return AgentSpec("writer", "test agent", policy, **kw)


def crash_at(event: str, step: int):
    fired = {"done": False}

    def hook(ev, idx, call):
        if not fired["done"] and ev == event and idx == step:
            fired["done"] = True
            raise SimulatedCrash(f"{ev} at step {idx}")
    return hook


# ---------------------------------------------------------------- sandbox

def test_tool_outside_policy_is_denied_and_reported_to_model(tmp_path):
    h = Harness(workspace_tools(tmp_path), ScriptedModel([ToolCall("write_file", {"path": "x", "content": "y"})]))
    rid = h.start(agent(Policy({"read_file"})), "try to write")
    step = h.store.steps(rid)[0]
    assert step["status"] == "error" and "not allowed" in step["error"]
    assert not (tmp_path / "x").exists()
    assert h.metrics[rid].denials == 1


def test_path_traversal_is_denied(tmp_path):
    h = Harness(workspace_tools(tmp_path), ScriptedModel([ToolCall("read_file", {"path": "../../etc/passwd"})]))
    rid = h.start(agent(), "escape")
    assert "denied" in h.store.steps(rid)[0]["error"]


def test_model_only_sees_allowed_tools(tmp_path):
    seen = {}

    class Spy(ScriptedModel):
        def next_action(self, context, tools):
            seen["tools"] = {t["name"] for t in tools}
            return Finish("ok")

    Harness(workspace_tools(tmp_path), Spy([])).start(agent(Policy({"read_file"})), "look")
    assert seen["tools"] == {"read_file"}


def test_bad_arguments_rejected(tmp_path):
    h = Harness(workspace_tools(tmp_path), ScriptedModel([ToolCall("write_file", {"path": "a"})]))
    rid = h.start(agent(), "missing content")
    assert "invalid arguments" in h.store.steps(rid)[0]["error"]


# ---------------------------------------------------------------- retries

class N(BaseModel):
    n: int = 0


def flaky_registry(fail_times: int):
    reg = ToolRegistry()
    state = {"calls": 0}

    def flaky(n: int = 0):
        state["calls"] += 1
        if state["calls"] <= fail_times:
            raise RetryableToolError("rate limited")
        return "ok"
    reg.register(Tool("flaky", "sometimes fails", N, flaky))
    return reg, state


def test_transient_errors_are_retried_with_backoff():
    reg, state = flaky_registry(2)
    sleeps = []
    h = Harness(reg, ScriptedModel([ToolCall("flaky", {})]), sleep=sleeps.append)
    rid = h.start(agent(Policy({"flaky"}), max_retries=2, backoff_s=0.1), "go")
    s = h.store.steps(rid)[0]
    assert s["status"] == "done" and s["attempts"] == 3 and state["calls"] == 3
    assert sleeps == [0.1, 0.2]


def test_retry_budget_is_bounded():
    reg, state = flaky_registry(10)
    h = Harness(reg, ScriptedModel([ToolCall("flaky", {})]), sleep=lambda s: None)
    rid = h.start(agent(Policy({"flaky"}), max_retries=2), "go")
    assert state["calls"] == 3 and "gave up after 3" in h.store.steps(rid)[0]["error"]


# ---------------------------------------------------------------- crash and resume

SCRIPT = [
    ToolCall("write_file", {"path": "report.md", "content": "# Report"}),
    ToolCall("append_line", {"path": "audit.log", "line": "report created"}),
    ToolCall("write_file", {"path": "summary.md", "content": "done"}),
    ToolCall("append_line", {"path": "audit.log", "line": "summary created"}),
]


def test_resume_replays_completed_steps_without_re_executing(tmp_path):
    db = RunStore(tmp_path / "runs.db")
    h1 = Harness(workspace_tools(tmp_path / "ws"), ScriptedModel(SCRIPT), db, fault_hook=crash_at("before_tool", 2))
    with pytest.raises(SimulatedCrash):
        h1.start(agent(), "make report", run_id="r1")

    h2 = Harness(workspace_tools(tmp_path / "ws"), ScriptedModel(SCRIPT), RunStore(tmp_path / "runs.db"))
    h2.resume(agent(), "r1")
    assert h2.store.get_run("r1")["status"] == "done"
    assert (tmp_path / "ws/audit.log").read_text().splitlines() == ["report created", "summary created"]
    assert h2.metrics["r1"].replayed_from_log == 2


def test_interrupted_non_idempotent_step_stops_for_review(tmp_path):
    db = tmp_path / "runs.db"
    h1 = Harness(workspace_tools(tmp_path / "ws"), ScriptedModel(SCRIPT), RunStore(db), fault_hook=crash_at("after_tool", 1))
    with pytest.raises(SimulatedCrash):
        h1.start(agent(), "make report", run_id="r2")

    h2 = Harness(workspace_tools(tmp_path / "ws"), ScriptedModel(SCRIPT), RunStore(db))
    h2.resume(agent(), "r2")
    run = h2.store.get_run("r2")
    assert run["status"] == "needs_review" and "append_line" in run["answer"]
    assert (tmp_path / "ws/audit.log").read_text().splitlines() == ["report created"]  # not duplicated


def test_crash_before_dispatch_of_non_idempotent_step_resumes_safely(tmp_path):
    db = tmp_path / "runs.db"
    h1 = Harness(workspace_tools(tmp_path / "ws"), ScriptedModel(SCRIPT), RunStore(db), fault_hook=crash_at("before_tool", 1))
    with pytest.raises(SimulatedCrash):
        h1.start(agent(), "make report", run_id="r4")
    assert RunStore(db).steps("r4")[-1]["status"] == "started"  # recorded intent, never dispatched
    h2 = Harness(workspace_tools(tmp_path / "ws"), ScriptedModel(SCRIPT), RunStore(db))
    h2.resume(agent(), "r4")
    assert h2.store.get_run("r4")["status"] == "done"
    assert (tmp_path / "ws/audit.log").read_text().splitlines() == ["report created", "summary created"]


def test_interrupted_idempotent_step_is_safely_repeated(tmp_path):
    db = tmp_path / "runs.db"
    h1 = Harness(workspace_tools(tmp_path / "ws"), ScriptedModel(SCRIPT), RunStore(db), fault_hook=crash_at("after_tool", 2))
    with pytest.raises(SimulatedCrash):
        h1.start(agent(), "make report", run_id="r3")
    h2 = Harness(workspace_tools(tmp_path / "ws"), ScriptedModel(SCRIPT), RunStore(db))
    h2.resume(agent(), "r3")
    assert h2.store.get_run("r3")["status"] == "done"
    assert (tmp_path / "ws/audit.log").read_text().splitlines() == ["report created", "summary created"]


def test_max_steps_marks_run_failed(tmp_path):
    loop = [ToolCall("list_files", {})] * 10
    h = Harness(workspace_tools(tmp_path), ScriptedModel(loop))
    rid = h.start(agent(max_steps=3), "loop forever")
    assert h.store.get_run(rid)["status"] == "failed"


# ---------------------------------------------------------------- context budget

def test_context_compacts_old_results_and_keeps_recent():
    steps = [{"status": "done", "tool": "read_file", "args": {"path": f"f{i}"}, "result": "x" * 2000, "error": None}
             for i in range(10)]
    msgs, compacted = build_context("sys", "task", steps, ContextBudget(max_tokens=2500, keep_recent_steps=2))
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert compacted > 0
    assert not tool_msgs[-1].get("compacted") and not tool_msgs[-2].get("compacted")
    assert tool_msgs[0]["content"].startswith("[compacted]")


def test_interrupted_step_result_is_hidden_from_model():
    steps = [{"status": "started", "tool": "append_line", "args": {}, "result": None, "error": None}]
    msgs, _ = build_context("sys", "task", steps, ContextBudget())
    assert [m["role"] for m in msgs] == ["system", "user"]


# ---------------------------------------------------------------- skills

def test_distill_writes_skill_after_repeated_success_and_loads_it(tmp_path):
    store = RunStore()
    script = [ToolCall("write_file", {"path": "a", "content": "1"}), ToolCall("read_file", {"path": "a"})]
    h = Harness(workspace_tools(tmp_path / "ws"), ScriptedModel(script), store)
    for _ in range(2):
        h.model = ScriptedModel(script)
        h.start(agent(), "make and check a file", task_kind="make_file")
    index = distill(store, tmp_path / "skills", min_support=2)
    assert index["make_file"]["steps"] == ["write_file", "read_file"]
    assert (tmp_path / "skills/make_file.md").exists()
    assert "write_file -> read_file" in load_skills_index(tmp_path / "skills", "make_file")


def test_distill_ignores_one_offs(tmp_path):
    store = RunStore()
    h = Harness(workspace_tools(tmp_path / "ws"), ScriptedModel([ToolCall("list_files", {})]), store)
    h.start(agent(), "once", task_kind="rare")
    assert "rare" not in distill(store, tmp_path / "skills", min_support=2)


# ---------------------------------------------------------------- API and MCP

def test_api_with_injected_model(tmp_path):
    from fastapi.testclient import TestClient
    from harness.api import create_app

    app = create_app(ScriptedModel([ToolCall("list_files", {})]), RunStore(), tmp_path / "ws", tmp_path / "skills")
    c = TestClient(app)
    assert c.get("/healthz").json()["model"] is True
    r = c.post("/runs", json={"agent": "reader", "task": "what is here?"}).json()
    assert r["status"] == "done" and r["steps"][0]["tool"] == "list_files"
    assert c.get(f"/runs/{r['id']}").status_code == 200
    assert c.post("/runs", json={"agent": "nope", "task": "x"}).status_code == 404


def test_api_without_model_refuses_new_runs(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from harness.api import create_app

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = TestClient(create_app(None, RunStore(), tmp_path / "ws", tmp_path / "skills"))
    assert c.post("/runs", json={"task": "x"}).status_code == 503


def test_mcp_bridge_exposes_only_allowed_tools(tmp_path):
    import asyncio
    from harness.mcp_bridge import build_server

    s = build_server(workspace_tools(tmp_path), Policy({"read_file", "list_files"}))
    assert {t.name for t in asyncio.run(s.list_tools())} == {"read_file", "list_files"}
