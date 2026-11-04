"""Kill a real process in the middle of applying an approved triage fix, then recover it.

Every step below is a separate Python process. The crash is os._exit(137), the same as kill -9:
no exception handlers run, nothing is flushed. All that survives is what was already in SQLite.

    python -m triage_bridge.demo
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

CASE = "SYN-007"   # schema change upstream breaks sp_fill_rate; downstream table and report go stale


def step(title: str, workdir: str, cmd: str, crash: str | None = None) -> int:
    print(f"\n$ {title}")
    env = dict(os.environ)
    env.pop("TRIAGE_CRASH", None)
    if crash:
        env["TRIAGE_CRASH"] = crash
    r = subprocess.run([sys.executable, "-m", "triage_bridge.cli", workdir, cmd, CASE], env=env,
                       capture_output=True, text=True)
    print(r.stdout.rstrip())
    if r.returncode not in (0, 3, 137):
        print(r.stderr)
    return r.returncode


def main() -> None:
    d = tempfile.mkdtemp(prefix="triage_in_harness_")
    naive = d + "_restart"
    print(f"Incident {CASE}, workdir {d}")
    step("1. triage diagnoses the incident and pauses at the human gate", d, "diagnose")
    rc = step("2. a person approves; the harness starts applying the fix ... and the process is killed", d, "approve",
              crash="rerun_job:after_tool")
    print(f"  (exit code {rc})")
    shutil.copytree(d, naive)
    step("3. what survived the crash", d, "show")
    step("4. a NEW process recovers from the logs", d, "recover")
    step("5. a person checks that one step, and the run finishes", d, "resolve")
    step("6. end state", d, "show")

    print("\n$ for comparison: the same crash, recovered by running the approved plan again from the top")
    sys.path.insert(0, os.getcwd())
    from harness import Harness, RunStore, ScriptedModel
    from .bench import _agent
    from .runner import TriageInHarness, plan_to_calls
    t = TriageInHarness(naive)
    v = t.service.view(CASE)
    calls = plan_to_calls(t.dag, CASE, v["category"], v["plan"], v.get("proposal") or "")
    Harness(t.warehouse.tools(), ScriptedModel(calls), RunStore()).start(_agent(len(calls)), "re-apply", run_id="restart")
    c = t.warehouse.counts()
    print(f"  warehouse: job runs {c['job_runs']}, ticket comments {c['comments']}, "
          f"audit lines {sum(c['audit_lines'].values())}, duplicated actions {t.warehouse.duplicated_actions()}")


if __name__ == "__main__":
    main()
