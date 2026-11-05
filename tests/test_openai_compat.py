"""OpenAICompatModel: request shape and response parsing, against a fake server (no network)."""
import json

import httpx

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
