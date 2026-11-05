"""Did the incident actually get resolved, or did the plan just run?

Everything else in this repo measures whether the approved actions happened: the right ones, once
each, nothing extra. That is a property of the runtime and it is the thing Keel is responsible for.
It is not the same question as whether the report the on-call engineer complained about came back.

A plan can be applied perfectly and still leave the pipeline broken: aimed at the wrong node, or at
the right node with a downstream table left out, or in an order that rebuilds a table before the job
that feeds it. Nothing in the run log can tell you that, because the log only knows what was called.

So this replays the recorded actions against the pipeline's structure and works out what state each
node ended in:

    fresh   it has current data
    stale   its data is old because something it reads from is not fresh
    failed  it is broken and has not been repaired and re-run

Two rules do the work, and both come from the pipeline rather than from the plan. A broken node is
only repaired by a run once the reason it broke is gone: a procedure that errors needs its code
patched first, because running it again runs the same bad code, while a load that never ran needs
nothing but a run. And anything run or refreshed is only as fresh as its inputs, so rebuilding a
report while its source is still stale leaves it stale. The incident is resolved when the node the
person complained about is fresh again.

The starting state is derived from the failure rather than read off the incident. An incident marks
every node below the break stale, which is the symptom spreading, not independent decay: a table on
an untouched branch of the pipeline is not part of this incident and should not have to be refreshed
for the fix to count.

This is a model of a pipeline, not a pipeline, and it is honest about what it can catch: a wrong
target, a missing downstream refresh, a wrong order. It cannot tell you whether the SQL in a patch
was correct, because a patch here is a line of text rather than code that runs. That needs a real
warehouse behind `Executor` and `DataProbe`, which is the production scope in the README.
"""
from __future__ import annotations

from triage.dag import PipelineDAG

from .warehouse import Warehouse

FRESH, STALE, FAILED = "fresh", "stale", "failed"


def broken(incident: dict) -> dict[str, bool]:
    """The nodes that are not working, each mapped to whether its code has to change first.

    A procedure reported "failed" errors on the data it is given, so a patch has to land before a
    run will do anything. A load reported "missing" simply did not run.
    """
    out: dict[str, bool] = {}
    for node, s in (incident.get("node_status") or {}).items():
        status = (s or {}).get("status")
        if status == "failed":
            out[node] = True
        elif status == "missing":
            out[node] = False
    return out


def initial(dag: PipelineDAG, incident: dict) -> dict[str, str]:
    """What state every node was in when the incident was filed, derived from what broke.

    A node reading from something broken is stale. Everything else is fresh, including a load
    reported as "changed": renaming a column does not stop it having current data, and the damage
    shows up in whatever reads the old name, which is what makes these incidents hard.
    """
    bad = broken(incident)
    return {n: FAILED if n in bad else STALE if any(a in bad for a in dag.upstream(n)) else FRESH
            for n in dag.nodes}


def health(dag: PipelineDAG, incident: dict, warehouse: Warehouse) -> dict[str, str]:
    """Replay what was done, in the order it was done, and return where every node ended up."""
    state = initial(dag, incident)
    needs_patch = {n for n, patch in broken(incident).items() if patch}
    patched: set[str] = set()
    for action, node in warehouse.actions():
        if action == "patch_procedure":
            patched.add(node)                      # a patch alone changes no data; it has to be re-run
            continue
        if node not in dag.nodes:
            continue
        if state[node] == FAILED and node in needs_patch and node not in patched:
            continue                               # re-running broken code runs the same broken code
        state[node] = FRESH if all(state[p] == FRESH for p in dag.parents[node]) else STALE
    return state


def unresolved(dag: PipelineDAG, incident: dict, warehouse: Warehouse) -> list[str]:
    """Nodes still not fresh, symptom first. Empty means the pipeline is healthy again."""
    state = health(dag, incident, warehouse)
    symptom = incident.get("symptom_node")
    return sorted((n for n in dag.nodes if state[n] != FRESH), key=lambda n: (n != symptom, n))


def resolved(dag: PipelineDAG, incident: dict, warehouse: Warehouse) -> bool:
    """The question the person who filed the incident actually asked: is the report right again?"""
    symptom = incident.get("symptom_node")
    return symptom is None or health(dag, incident, warehouse)[symptom] == FRESH


def explain(dag: PipelineDAG, incident: dict, warehouse: Warehouse) -> str:
    state = health(dag, incident, warehouse)
    symptom = incident.get("symptom_node")
    if symptom is None:
        return "the incident names no node to check"
    if state[symptom] == FRESH:
        return f"{symptom} is fresh again"
    blockers = [n for n in dag.upstream(symptom) if state[n] != FRESH]
    because = f", because {', '.join(f'{n} is {state[n]}' for n in blockers)}" if blockers else ""
    return f"{symptom} is still {state[symptom]}{because}"
