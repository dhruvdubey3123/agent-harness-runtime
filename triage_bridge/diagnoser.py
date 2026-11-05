"""The first of the two agents: the triage investigator, run under the harness.

The investigator itself lives in pipeline-triage-graph (triage/investigator.py): the prompt, the five
read-only tools over the incident and the pipeline, and the checks on what it submits. That project
owns what a good diagnosis is. This module owns only how it runs here: the same tools wrapped into
the harness sandbox, so every call is in the crash-safe run log, the agent can be killed and resumed
by a new process, and a valid submit_diagnosis ends the run.

Its sandbox has no tool that changes anything, so the worst case is a wrong answer rather than a
wrong action, and a wrong answer still has to get past the reviewer and then a person.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from triage.dag import PipelineDAG
from triage.investigator import INSTRUCTIONS, TERMINAL, derive_plan, task
from triage.investigator import tools as investigator_tools
from triage.knowledge import CaseLibrary

from harness import AgentSpec, Policy, ToolRegistry

READ_ONLY = {"pipeline", "status", "upstream", "precedents", TERMINAL}

__all__ = ["READ_ONLY", "INSTRUCTIONS", "Diagnosis", "derive_plan", "spec", "task", "tools"]


@dataclass
class Diagnosis:
    category: str
    fix_target: str
    cause: str = ""
    reason: str = ""
    plan: list[str] = field(default_factory=list)


def tools(dag: PipelineDAG, library: CaseLibrary, incident: dict) -> ToolRegistry:
    """The investigator's tools, unchanged, registered as read-only and safe to repeat. The call itself
    is the record: the next process reads the diagnosis back out of the run log."""
    reg = ToolRegistry()
    for t in investigator_tools(dag, library, incident):
        reg.tool(t.name, t.description, t.args, idempotent=True, side_effects=False)(t.fn)
    return reg


def spec(max_steps: int = 14) -> AgentSpec:
    return AgentSpec("diagnoser", INSTRUCTIONS, Policy(READ_ONLY, max_calls_per_tool=6), max_steps=max_steps,
                     finish_on=frozenset({TERMINAL}))
