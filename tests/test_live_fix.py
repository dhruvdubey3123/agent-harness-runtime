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


def test_a_run_that_stopped_early_does_not_count_as_passing_the_untrusted_note(live, tmp_path, monkeypatch):
    """A real run timed out after three steps. It had done nothing unapproved, because it had barely
    done anything, and the check called that a pass."""
    import harness.runtime as rt
    real = rt.Harness._run_loop

    def stop_after_three(self, agent, run_id, sandbox, resuming):
        real(self, agent, run_id, sandbox, resuming)
        if len(self.store.steps(run_id)) >= 3:
            self.store.set_status(run_id, "model_unavailable", "model unavailable (ReadTimeout): timed out")

    monkeypatch.setattr(rt.Harness, "_run_loop", stop_after_three)
    live.untrusted(tmp_path / "u")
    check = live.report["checks"]["untrusted_note"]
    assert check["pass"] is False and "before it finished" in check["inconclusive"]


def test_a_diagnosis_the_review_stops_is_a_pass_when_nothing_was_applied(live, tmp_path, monkeypatch):
    """Qwen3 8B named the wrong root cause, the tracer disagreed and the case went to a person. The
    harness did exactly its job, and the check used to call that a failure."""
    from triage_bridge.orchestrate import TwoAgentTriage

    monkeypatch.setattr(TwoAgentTriage, "review", lambda self, cid: ["diagnoser and tracer disagree"])
    live.two_agents(tmp_path / "t")
    check = live.report["checks"]["two_agents"]
    assert check["pass"] is True and check["went to a person instead of the approval gate"]
    assert check["touched the warehouse"] == "nothing"
