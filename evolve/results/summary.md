# Recovery policy search

Proposer: `rules`. 10 configurations measured on the same 75 crashed fixes; 4 rejected by the gate.

| configuration | unattended | duplicated | dropped | verdict |
|---|---|---|---|---|
| `[all tools pause] retries=2 gate>=0.5` | 45/75 | 0 | 0 | passed |
| `[rerun_job=unique_payload] retries=2 gate>=0.5` | 57/75 | 0 | 0 | passed |
| `[comment_ticket=unique_payload, rerun_job=unique_payload] retries=2 gate>=0.5` | 67/75 | 0 | 0 | passed |
| `[append_audit=unique_payload, comment_ticket=unique_payload, rerun_job=unique_payload] retries=2 gate>=0.5` | 75/75 | 0 | 0 | kept |
| `[append_audit=unique_payload, comment_ticket=unique_payload] retries=2 gate>=0.5` | 63/75 | 0 | 0 | passed |
| `[append_audit=unique_payload, comment_ticket=unique_payload, rerun_job=assume_done] retries=2 gate>=0.5` | 0/0 | 0 | 0 | rejected: probe repeated_action_never_ran -> dropped |
| `[append_audit=unique_payload, comment_ticket=unique_payload, rerun_job=retry] retries=2 gate>=0.5` | 0/0 | 0 | 0 | rejected: probe repeated_action_already_ran -> duplicated |
| `[append_audit=unique_payload, rerun_job=unique_payload] retries=2 gate>=0.5` | 65/75 | 0 | 0 | passed |
| `[append_audit=unique_payload, comment_ticket=assume_done, rerun_job=unique_payload] retries=2 gate>=0.5` | 75/75 | 0 | 5 | rejected: invariant no_dropped_actions; invariant every_run_exactly_once |
| `[append_audit=unique_payload, comment_ticket=retry, rerun_job=unique_payload] retries=2 gate>=0.5` | 75/75 | 5 | 0 | rejected: invariant no_duplicated_actions; invariant every_run_exactly_once |

**Result.** Unattended recovery went from 45/75 to 75/75 (60% to 100%). Duplicated actions stayed at 0 and dropped actions at 0; a candidate that changed either was thrown away.

**Frozen during the search:**
- the tool allowlist each agent gets (harness.Policy)
- which tools are marked idempotent (triage_bridge.warehouse)
- the human approval gate and its position in the graph
- the benchmark, its incidents and their labels
- the safety invariants and the counterexample probes in evolve.guard
