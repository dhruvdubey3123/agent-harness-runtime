"""OpenAICompatModel: request shape and response parsing, against a fake server (no network)."""
import json

import httpx
import pytest

from harness import Finish, OpenAICompatModel, ToolCall


class FakeResponse:
    status_code, headers = 200, {}

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_tool_call_round_trip(monkeypatch):
    seen = {}

    def fake_post(url, json=None, timeout=None, headers=None):
        seen["url"], seen["body"] = url, json
        return FakeResponse({"choices": [{"message": {"tool_calls": [{"id": "a", "type": "function", "function": {
            "name": "read_file", "arguments": '{"path": "incident.md"}'}}]}}]})

    monkeypatch.setattr(httpx, "post", fake_post)
    m = OpenAICompatModel(model="qwen3:8b", base_url="http://localhost:11434/v1")
    ctx = [{"role": "system", "content": "be careful"}, {"role": "user", "content": "do it"},
           {"role": "assistant", "tool": "list_files", "args": {}}, {"role": "tool", "content": "['incident.md']"}]
    tools = [{"name": "read_file", "description": "read", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}]
    action = m.next_action(ctx, tools)
    assert action == ToolCall("read_file", {"path": "incident.md"})
    assert seen["url"] == "http://localhost:11434/v1/chat/completions"
    msgs = seen["body"]["messages"]
    assert [x["role"] for x in msgs] == ["system", "user", "assistant", "tool"]
    assert msgs[2]["tool_calls"][0]["id"] == msgs[3]["tool_call_id"]
    assert json.loads(msgs[2]["tool_calls"][0]["function"]["arguments"]) == {}
    assert seen["body"]["tools"][0]["function"]["parameters"]["properties"]["path"]["type"] == "string"


def test_final_answer_strips_thinking(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResponse(
        {"choices": [{"message": {"content": "<think>let me see</think>\nAll done."}}]}))
    assert OpenAICompatModel().next_action([{"role": "user", "content": "x"}], []) == Finish("All done.")


def test_unparseable_arguments_reach_the_sandbox_as_an_error(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResponse({"choices": [{"message": {"tool_calls": [
        {"function": {"name": "write_file", "arguments": "{path: oops"}}]}}]}))
    action = OpenAICompatModel().next_action([{"role": "user", "content": "x"}], [])
    assert action.tool == "write_file" and "_unparsed" in action.args


def test_a_rate_limited_free_tier_is_waited_out(monkeypatch):
    import harness.model as hm
    replies = [429, 200]

    class R(FakeResponse):
        def __init__(self, code):
            super().__init__({"choices": [{"message": {"content": "done"}}]})
            self.status_code, self.headers = code, {"retry-after": "0"}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: R(replies.pop(0)))
    monkeypatch.setattr(hm.time, "sleep", lambda s: None)
    assert OpenAICompatModel().next_action([{"role": "user", "content": "x"}], []) == Finish("done")


def test_output_the_provider_could_not_parse_is_asked_again(monkeypatch):
    """Groq answers 400 when it cannot turn gpt-oss-120b's output into a tool call. Nothing ran, so the
    question is asked again rather than pausing the run."""
    import harness.model as hm
    replies = [400, 200]

    class R(FakeResponse):
        text = '{"error":{"message":"Parsing failed. The model generated output that could not be parsed."}}'

        def __init__(self, code):
            super().__init__({"choices": [{"message": {"content": "recovered"}}]})
            self.status_code, self.headers = code, {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("err", request=None, response=self)

    monkeypatch.setattr(httpx, "post", lambda *a, **k: R(replies.pop(0)))
    monkeypatch.setattr(hm.time, "sleep", lambda s: None)
    assert OpenAICompatModel().next_action([{"role": "user", "content": "x"}], []) == Finish("recovered")


GROQ_429 = ('{"error":{"message":"Rate limit reached for model `openai/gpt-oss-120b` ... on tokens per minute '
            '(TPM): Limit 8000, Used 7500, Requested 1200. Please try again in 7m2.5s."}}')


def rate_limited(text=GROQ_429, headers=None):
    class R(FakeResponse):
        def __init__(self):
            super().__init__({"choices": [{"message": {"content": "ok"}}]})
            self.status_code, self.headers, self.text = 429, headers or {}, text

        def raise_for_status(self):
            raise httpx.HTTPStatusError("429", request=None, response=self)

    return R()


def test_the_providers_own_wait_is_honoured_not_capped_at_a_guess(monkeypatch):
    """Groq asks for minutes when the per-minute budget is gone. Sleeping 60s six times and failing
    anyway wasted the retries and hid the reason."""
    import harness.model as hm
    slept = []
    monkeypatch.setattr(hm.time, "sleep", slept.append)
    calls = [rate_limited(), FakeResponse({"choices": [{"message": {"content": "through"}}]})]
    monkeypatch.setattr(httpx, "post", lambda *a, **k: calls.pop(0))
    monkeypatch.setenv("HARNESS_MAX_WAIT", "600")
    assert OpenAICompatModel().next_action([{"role": "user", "content": "x"}], []) == Finish("through")
    assert slept == [422.5]


def test_a_wait_longer_than_we_will_sit_through_fails_with_the_reason(monkeypatch):
    import harness.model as hm
    monkeypatch.setattr(hm.time, "sleep", lambda s: pytest.fail("should not sleep"))
    monkeypatch.setattr(httpx, "post", lambda *a, **k: rate_limited())
    monkeypatch.setenv("HARNESS_MAX_WAIT", "60")
    with pytest.raises(httpx.HTTPStatusError) as e:
        OpenAICompatModel().next_action([{"role": "user", "content": "x"}], [])
    assert "tokens per minute" in str(e.value)


def test_the_retry_after_header_wins_when_it_is_there(monkeypatch):
    import harness.model as hm
    slept = []
    monkeypatch.setattr(hm.time, "sleep", slept.append)
    calls = [rate_limited(text="", headers={"retry-after": "12"}),
             FakeResponse({"choices": [{"message": {"content": "through"}}]})]
    monkeypatch.setattr(httpx, "post", lambda *a, **k: calls.pop(0))
    assert OpenAICompatModel().next_action([{"role": "user", "content": "x"}], []) == Finish("through")
    assert slept == [12.0]


def test_a_request_bigger_than_the_whole_budget_fails_at_once(monkeypatch):
    """Waiting cannot make one request fit inside a per-minute limit smaller than the request."""
    import harness.model as hm
    monkeypatch.setattr(hm.time, "sleep", lambda s: pytest.fail("waiting would never help"))
    monkeypatch.setattr(httpx, "post", lambda *a, **k: rate_limited(
        text='{"error":{"message":"Rate limit reached ... (TPM): Limit 8000, Used 1200, Requested 9400."}}'))
    with pytest.raises(httpx.HTTPStatusError) as e:
        OpenAICompatModel().next_action([{"role": "user", "content": "x"}], [])
    assert "Requested 9400" in str(e.value)
