"""Run the pipeline-triage-graph agent inside the harness.

Triage decides *what* to fix and a person approves it. The harness then applies the approved
plan as a durable run: every warehouse action is logged before and after it happens, so a crash
in the middle of a fix resumes without re-running a load or posting the same ticket comment twice.

    pip install -e ".[triage]"
    python -m triage_bridge.demo        # real process kill in the middle of a fix, then recovery
    python -m triage_bridge.bench       # every approvable incident in the eval set, crashed and recovered
"""
from .runner import NeedsReview, TriageInHarness
from .warehouse import Warehouse

__all__ = ["NeedsReview", "TriageInHarness", "Warehouse"]
