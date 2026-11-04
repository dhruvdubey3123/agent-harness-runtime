"""The real-model path, driven over HTTP by a stub instead of by a model.

triage_bridge.live_fix is meant to be run against a local Ollama, which CI does not have. The
part worth guarding there is not the model's competence but the plumbing and the three claims the
script makes, so the same script runs here against a server that returns a fixed sequence of tool
calls, two of which nobody approved.
"""
from __future__ import annotations

import importlib

import pytest

pytest.importorskip("triage")
pytest.importorskip("langgraph.checkpoint.sqlite")

from stub_model_server import serving  # noqa: E402  (sibling module: pytest puts tests/ on the path)


@pytest.fixture
def live(monkeypatch):
    with serving() as url:
        monkeypatch.setenv("HARNESS_BASE_URL", url)
        monkeypatch.setenv("HARNESS_MODEL", "stub-fixer")
        monkeypatch.setenv("NO_PROXY", "*")
        mod = importlib.reload(importlib.import_module("triage_bridge.live_fix"))
        mod.preflight()
        yield mod


def test_every_check_in_the_live_script_holds(live, tmp_path):
    live.clean(tmp_path / "clean")
    live.crash_and_resume(tmp_path / "crash")
    live.untrusted(tmp_path / "untrusted")
    live.two_agents(tmp_path / "two_agents")
    assert [name for name, c in live.report["checks"].items() if not c["pass"]] == []
    assert live.report["checks"]["two_agents"]["root cause correct"]


def test_the_stub_really_does_try_to_exceed_the_plan(live, tmp_path):
    """If this ever comes back empty, the script above is passing because nothing was attempted."""
    live.untrusted(tmp_path / "untrusted")
    tried = live.report["checks"]["untrusted_note"]["unapproved calls it tried"]
    assert len(tried) == 2 and all("none" not in t for t in tried)
    counts = live.report["checks"]["untrusted_note"]
    assert "load_cases" not in counts["job_runs"] and counts["comments"] == {"SYN-007": 1}
