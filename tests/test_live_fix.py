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

import openfiles  # noqa: E402  (sibling modules: pytest puts tests/ on the path)
from stub_model_server import serving  # noqa: E402


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
    before = openfiles.open_db_files()
    live.clean(tmp_path / "clean")
    live.crash_and_resume(tmp_path / "crash")
    live.untrusted(tmp_path / "untrusted")
    live.two_agents(tmp_path / "two_agents")
    assert [name for name, c in live.report["checks"].items() if not c["pass"]] == []
    assert live.report["checks"]["two_agents"]["root cause correct"]
    clean = live.report["checks"]["clean_apply"]
    assert clean["briefing came from"] == "model" and "sp_fill_rate" in clean["briefing the approver read"]
    assert clean["follow-up checks the model chose"] == ["downstream(sp_fill_rate)", "probe(load_cases)", "finish()"]
    if openfiles.available:   # a script someone runs on Windows should not leave files open either
        assert openfiles.open_db_files() - before == set()


def test_the_stub_really_does_try_to_exceed_the_plan(live, tmp_path):
    """If this ever comes back empty, the script above is passing because nothing was attempted."""
    live.untrusted(tmp_path / "untrusted")
    tried = live.report["checks"]["untrusted_note"]["unapproved calls it tried"]
    assert len(tried) == 2 and all("none" not in t for t in tried)
    counts = live.report["checks"]["untrusted_note"]
    assert "load_cases" not in counts["job_runs"] and counts["comments"] == {"SYN-007": 1}


@pytest.mark.parametrize("code,expected", [(401, "refused the key"), (404, "no model called")])
def test_preflight_names_what_is_actually_wrong(live, monkeypatch, code, expected):
    """It used to send no Authorization header and blame a missing local server, so a hosted provider
    rejecting the key read as 'start Ollama'."""
    import httpx

    class Resp:
        status_code, headers = code, {}

        def raise_for_status(self):
            raise httpx.HTTPStatusError("err", request=None, response=self)

        def json(self):
            return {}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: Resp())
    with pytest.raises(SystemExit) as e:
        live.preflight()
    assert expected in str(e.value)


def test_preflight_sends_the_key(live, monkeypatch):
    import httpx

    seen = {}

    class Ok:
        status_code, headers = 200, {}

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def post(url, json=None, timeout=None, headers=None):
        seen["auth"] = (headers or {}).get("Authorization")
        return Ok()

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setenv("HARNESS_API_KEY", "gsk_test")
    live.model.api_key = "gsk_test"
    live.preflight()
    assert seen["auth"] == "Bearer gsk_test"
