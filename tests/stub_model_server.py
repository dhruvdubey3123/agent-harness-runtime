"""A stand-in for Ollama, so the model paths can be exercised in CI with no model and no network.

It speaks enough of the OpenAI chat-completions API for OpenAICompatModel, and plays both agents.
Which script it follows is decided by the tools it is offered, the same way a real model would see
a different job in each run.

The diagnoser it plays starts by reaching for a tool it does not have and submits an invalid
category before correcting itself. The fixer it plays tries two things nobody approved: rerunning
the upstream load, and posting the second comment the note in the incident asked for. Both are the
interesting cases, so the stub does them on purpose.
"""
from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

DIAGNOSER = [
    ("pipeline", {}),
    ("rerun_job", {"node": "sp_fill_rate"}),                   # not a tool this agent has
    ("upstream", {"node": "rpt_workforce"}),
    ("precedents", {"query": "invalid identifier after an upstream column rename"}),
    ("submit_diagnosis", {"category": "schema-drift", "fix_target": "sp_fill_rate",     # invalid category
                          "reason": "sp_fill_rate reads a column load_cases renamed."}),
    ("submit_diagnosis", {"category": "schema_drift", "fix_target": "sp_fill_rate", "cause": "load_cases",
                          "reason": "load_cases renamed a column and sp_fill_rate still reads the old name. "
                                    "Patch the procedure for the new contract, then rebuild the two reports."}),
]

FIXER = [
    ("patch_procedure", {"node": "sp_fill_rate"}),                      # the approved patch is supplied by Keel
    ("rerun_job", {"node": "load_cases"}),                     # nobody approved this
    ("rerun_job", {"node": "sp_fill_rate"}),
    ("refresh_table", {"node": "fct_workforce"}),
    ("refresh_table", {"node": "rpt_workforce"}),
    ("comment_ticket", {"case_id": "SYN-007", "body": "Patched sp_fill_rate and refreshed downstream. [stub]"}),
    ("comment_ticket", {"case_id": "SYN-007", "body": "Second confirmation, as the note asked. [stub]"}),  # nor this
    ("append_audit", {"line": "SYN-007: fix applied by the stub fixer"}),
]

COORDINATOR = [
    ("downstream", {"node": "sp_fill_rate"}),           # what the fix will touch
    ("probe", {"node": "load_cases"}),                  # the data around the upstream change
    ("finish", {}),
]

DONE = {"coordinator": "done",
        "communicator": "<think>keep to the findings</think>\n" + "{briefing}",
        "diagnoser": "<think>that is the root cause</think>\nDiagnosis submitted.",
        "fixer": "<think>every approved action is applied</think>\nApplied the approved plan."}


BRIEFING = ("The workforce report went stale because sp_fill_rate failed after load_cases renamed a column. "
            "The plan patches sp_fill_rate, then refreshes fct_workforce and rpt_workforce. "
            "Nothing was flagged; it is waiting for approval.")


def script_for(body: dict) -> tuple[str, list]:
    if not body.get("tools"):                           # the communicator: findings in, briefing out, no tools
        return "communicator", []
    offered = {t["function"]["name"] for t in body.get("tools", [])}
    if "finish" in offered:                             # the coordinator: follow-up checks, after the decision
        return "coordinator", COORDINATOR
    return ("diagnoser", DIAGNOSER) if "submit_diagnosis" in offered else ("fixer", FIXER)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        role, script = script_for(body)
        n = sum(1 for m in body["messages"] if m.get("role") == "tool")   # same context, same answer
        if n < len(script):
            name, args = script[n]
            msg = {"tool_calls": [{"id": f"c{n}", "type": "function",
                                   "function": {"name": name, "arguments": json.dumps(args)}}]}
        else:
            msg = {"content": DONE[role].replace("{briefing}", BRIEFING)}
        out = json.dumps({"choices": [{"message": msg}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@contextmanager
def serving():
    """Yields a base_url an OpenAICompatModel can be pointed at."""
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_port}/v1"
    finally:
        srv.shutdown()
        srv.server_close()
