"""HTTP control plane.  uvicorn harness.api:app

Runs need a real model, so POST /runs uses Claude when ANTHROPIC_API_KEY is set, an
OpenAI-compatible server such as Ollama when HARNESS_BASE_URL is set, and returns 503
otherwise. Everything else (inspecting runs, resuming, distilling skills) works without a key. Tests inject a ScriptedModel through `create_app`.
"""
from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .model import Model
from .runtime import AgentSpec, Harness
from .sandbox import Policy
from .skills import distill
from .store import RunStore
from .tools import workspace_tools

WORKSPACE = Path(os.getenv("HARNESS_WORKSPACE", "./workspace")).resolve()
SKILLS = Path(os.getenv("HARNESS_SKILLS", "./skills")).resolve()

AGENTS = {
    "reader": AgentSpec("reader", "Answer questions using only files in the workspace.",
                        Policy({"read_file", "list_files"})),
    "writer": AgentSpec("writer", "Complete file tasks in the workspace. Prefer write_file over append_line.",
                        Policy({"read_file", "list_files", "write_file", "append_line"})),
}


class RunRequest(BaseModel):
    agent: str = "reader"
    task: str
    task_kind: str | None = None


def create_app(model: Model | None = None, store: RunStore | None = None, workspace: Path = WORKSPACE,
               skills_dir: Path = SKILLS) -> FastAPI:
    workspace.mkdir(parents=True, exist_ok=True)
    store = store or RunStore(os.getenv("HARNESS_DB", ":memory:"))
    if model is None and os.getenv("ANTHROPIC_API_KEY"):
        from .model import AnthropicModel
        model = AnthropicModel()
    elif model is None and os.getenv("HARNESS_BASE_URL"):
        from .model import OpenAICompatModel   # Ollama, llama.cpp, vLLM: open-weight models, no key needed
        model = OpenAICompatModel()
    harness = Harness(workspace_tools(workspace), model, store, skills_dir=str(skills_dir)) if model else None
    app = FastAPI(title="agent-harness-runtime", version="0.1.0")

    def _agent(name: str) -> AgentSpec:
        if name not in AGENTS:
            raise HTTPException(404, f"unknown agent {name}; have {sorted(AGENTS)}")
        return AGENTS[name]

    def _view(run_id: str) -> dict:
        run = store.get_run(run_id)
        if not run:
            raise HTTPException(404, "unknown run")
        m = harness.metrics.get(run_id) if harness else None
        return {**run, "steps": store.steps(run_id), "metrics": asdict(m) if m else None}

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True, "model": harness is not None, "agents": sorted(AGENTS)}

    @app.post("/runs")
    def start(req: RunRequest) -> dict:
        if harness is None:
            raise HTTPException(503, "no model configured; set ANTHROPIC_API_KEY or HARNESS_BASE_URL")
        rid = harness.start(_agent(req.agent), req.task, req.task_kind)
        return _view(rid)

    @app.get("/runs/{run_id}")
    def get(run_id: str) -> dict:
        return _view(run_id)

    @app.post("/runs/{run_id}/resume")
    def resume(run_id: str) -> dict:
        if harness is None:
            raise HTTPException(503, "no model configured")
        run = store.get_run(run_id) or {}
        harness.resume(_agent(run.get("agent", "")), run_id)
        return _view(run_id)

    @app.post("/skills/distill")
    def do_distill(min_support: int = 2) -> dict:
        return distill(store, skills_dir, min_support)

    return app


app = create_app()
