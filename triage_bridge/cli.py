"""One step per process, so a crash really is a dead process.

    python -m triage_bridge.cli WORKDIR diagnose SYN-007
    python -m triage_bridge.cli WORKDIR approve SYN-007        # TRIAGE_CRASH=rerun_job:after_tool kills it mid-fix
    python -m triage_bridge.cli WORKDIR recover SYN-007
    python -m triage_bridge.cli WORKDIR resolve SYN-007
    python -m triage_bridge.cli WORKDIR show SYN-007
"""
from __future__ import annotations

import json
import os
import sys

from .runner import DATA, NeedsReview, TriageInHarness


def crash_hook():
    spec = os.environ.get("TRIAGE_CRASH")
    if not spec:
        return None
    tool, event = spec.split(":")

    def hook(ev, idx, call):
        if ev == event and call.tool == tool:
            print(f"  !! process killed (exit 137) during {tool}, window={event}", flush=True)
            os._exit(137)          # no cleanup, no except blocks: the same as kill -9
    return hook


def main(argv: list[str]) -> int:
    workdir, cmd, cid = argv[0], argv[1], argv[2]
    t = TriageInHarness(workdir, fault_hook=crash_hook())
    if cmd == "diagnose":
        inc = next(json.loads(l) for l in (DATA / "incidents.jsonl").read_text().splitlines() if f'"{cid}"' in l)
        v = t.diagnose(inc)
        print(f"  triage: root cause {v['fix_target']} ({v['category']}, confidence {v['confidence']})")
        print(f"  proposal: {v['proposal']}")
        print(f"  plan: {v['plan']}")
        print(f"  waiting for a person: {v['waiting_for_human']}")
    elif cmd == "approve":
        v = t.decide(cid, True, "on-call engineer")
        print(f"  outcome: {v.get('outcome')}")
    elif cmd in ("recover", "resolve"):
        try:
            if cmd == "recover":
                v = t.recover(cid)
            else:
                last = t.store.steps(f"apply-{cid}")[-1]
                landed = t.warehouse.took_effect(last["tool"], last["args"])
                print(f"  a person checks the warehouse: did {last['tool']}({last['args'].get('node') or last['args'].get('case_id')}) "
                      f"happen? {'yes' if landed else 'no'}")
                v = t.resolve(cid, landed)
            print(f"  outcome: {v.get('outcome')}")
        except NeedsReview as e:
            print(f"  harness stopped: {e.step['tool']}({e.step['args'].get('node') or e.step['args'].get('case_id')}) "
                  f"was {e.step['status']} when the process died and is not safe to repeat -> needs_review")
            return 3
    elif cmd == "show":
        st = t.apply_status(cid)
        if st:
            print(f"  harness run log ({st['status']}):")
            for s in st["steps"]:
                tgt = s["args"].get("node") or s["args"].get("case_id") or ""
                print(f"    {s['idx']} {s['tool']:<15} {tgt:<16} {s['status']}")
        c = t.warehouse.counts()
        print(f"  warehouse: job runs {c['job_runs']}, ticket comments {c['comments']}, "
              f"audit lines {sum(c['audit_lines'].values())}, duplicated actions {t.warehouse.duplicated_actions()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
