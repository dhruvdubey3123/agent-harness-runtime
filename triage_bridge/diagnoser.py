"""The first of the two agents: a model that works out what broke, with nothing it can break.

The triage graph finds the root cause by walking the pipeline structure, which is deterministic and
hard to beat on the incidents it was built for. This is the other way round: a model is given
read-only tools over the same incident and the same pipeline and has to find the cause itself, then
submit a diagnosis in a shape the rest of the system can check.

Its sandbox has no tool that changes anything, so the worst case is a wrong answer rather than a
wrong action, and a wrong answer still has to get past the reviewer and then a person. What it
submits is validated here: an unknown category, a node that is not in the pipeline, or a table
nobody is allowed to patch directly all come back as an error the model can see and correct.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from triage.dag import PipelineDAG, refresh_order
from triage.knowledge import CATEGORIES, CaseLibrary

from harness import AgentSpec, Policy, ToolRegistry
from harness.tools import NoArgs

READ_ONLY = {"pipeline", "status", "upstream", "precedents", "submit_diagnosis"}

INSTRUCTIONS = (
    "You are on call for a data pipeline. An incident says which table looks wrong. Find the node that "
    "actually broke, not the one that looks wrong: a stale report is usually the symptom of a failure "
    "further upstream. Use the tools to read the pipeline and the reported status of each node, and look "
    "for similar resolved cases. When you are confident, call submit_diagnosis once and stop. You cannot "
    "change anything from here, and nothing you submit is applied until a person approves it.\n"
    f"Categories: {', '.join(CATEGORIES)}."
)


class Submission(BaseModel):
    category: str = Field(description=f"one of: {', '.join(CATEGORIES)}")
    fix_target: str = Field(description="the node that has to be fixed")
    cause: str = Field("", description="the upstream node that caused it, if any")
    reason: str = Field(description="one or two sentences an on-call engineer can act on")


class NodeArg(BaseModel):
    node: str


class Query(BaseModel):
    query: str


@dataclass
class Diagnosis:
    category: str
    fix_target: str
    cause: str = ""
    reason: str = ""
    plan: list[str] = field(default_factory=list)


def derive_plan(dag: PipelineDAG, target: str) -> list[str]:
    """Derived, not proposed. Once the target is known, what else to re-run is pipeline structure,
    so there is no reason to let a model guess at it or to ask a person to check that it guessed
    right. The same derivation the rule-based orchestrator uses."""
    return [f"fix {target}", *[f"refresh {n}" for n in refresh_order(dag, target) if n != target]]


def tools(dag: PipelineDAG, library: CaseLibrary, incident: dict) -> ToolRegistry:
    reg = ToolRegistry()
    reported = incident.get("node_status", {})

    def one_line(node: str) -> str:
        st = reported.get(node)
        if st is None:
            return f"{node}: nothing reported"
        bits = [st.get("status", "?"), st.get("error", ""), st.get("note", ""), *st.get("changes", [])]
        return f"{node}: " + ", ".join(b for b in bits if b)

    @reg.tool("pipeline", "The pipeline: every node, its kind, and what it reads from.",
              NoArgs, idempotent=True, side_effects=False)
    def pipeline() -> str:
        lines = [f"pipeline {dag.name}, symptom reported at {incident['symptom_node']}: {incident.get('symptom', '')}"]
        for name, node in dag.nodes.items():
            parents = ", ".join(dag.parents[name]) or "none"
            protect = "  [protected: fix the job that produces it instead]" if name in dag.protected else ""
            lines.append(f"  {name} ({node.kind}) reads from: {parents}{protect}")
        return "\n".join(lines)

    @reg.tool("status", "What the incident reports about one node.", NodeArg, idempotent=True, side_effects=False)
    def status(node: str) -> str:
        if node not in dag.nodes:
            raise ValueError(f"{node} is not in this pipeline")
        return one_line(node)

    @reg.tool("upstream", "Every node this one depends on, nearest first, with its reported status.",
              NodeArg, idempotent=True, side_effects=False)
    def upstream(node: str) -> str:
        if node not in dag.nodes:
            raise ValueError(f"{node} is not in this pipeline")
        seen, frontier, order = {node}, list(dag.parents[node]), []
        while frontier:
            nxt = []
            for n in frontier:
                if n not in seen:
                    seen.add(n)
                    order.append(n)
                    nxt += list(dag.parents[n])
            frontier = nxt
        return "\n".join(one_line(n) for n in order) or "nothing upstream"

    @reg.tool("precedents", "Resolved cases that look similar, with how each was fixed.",
              Query, idempotent=True, side_effects=False)
    def precedents(query: str) -> str:
        hits = library.search(query, k=3)
        return "\n".join(f"{c.id} ({c.category}, score {s}): {c.summary} -> {c.resolution}"
                         for c, s in hits) or "no similar resolved cases"

    @reg.tool("submit_diagnosis", "Submit the diagnosis. Call this once, when you are confident.",
              Submission, idempotent=True, side_effects=False)
    def submit_diagnosis(category: str, fix_target: str, cause: str = "", reason: str = "") -> str:
        if category not in CATEGORIES:
            raise ValueError(f"'{category}' is not a category. Use one of: {', '.join(CATEGORIES)}")
        if fix_target not in dag.nodes:
            raise ValueError(f"'{fix_target}' is not a node in this pipeline. Call pipeline to see the nodes.")
        if fix_target in dag.protected:
            raise ValueError(f"{fix_target} is protected and cannot be patched directly. "
                             "Diagnose the job that produces it instead.")
        if cause and cause not in dag.nodes:
            raise ValueError(f"'{cause}' is not a node in this pipeline")
        # Nothing is kept in memory here: the call itself is the record, and the run log is where
        # the next process reads the diagnosis from.
        return f"recorded: fix {fix_target} ({category}), {len(derive_plan(dag, fix_target))} plan steps"

    return reg


def spec(max_steps: int = 14) -> AgentSpec:
    return AgentSpec("diagnoser", INSTRUCTIONS, Policy(READ_ONLY, max_calls_per_tool=6), max_steps=max_steps)


def task(incident: dict) -> str:
    return (f"Incident {incident.get('id')}: {incident.get('symptom')}, reported at "
            f"{incident['symptom_node']}. Find what actually broke and submit a diagnosis.")
