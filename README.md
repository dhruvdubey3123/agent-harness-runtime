# agent-harness-runtime

Most agent failures in production are not model failures. The model picked a reasonable next step,
and then the process died halfway through a tool call, or the context filled up with old logs, or the
agent called a tool it should never have seen. This repo is the part of an agent system that handles
those problems: the **harness**.

## In plain words

An AI agent runs a loop: decide, use a tool, look at the result, repeat. In real systems the program can crash
in the middle of that loop. If you restart from scratch, the agent may repeat actions that already happened,
such as sending a message twice or writing a duplicate log line. The harness is the safety layer around the model.

* It records every step before and after it runs.
* After a crash it picks up where it stopped instead of starting over.
* When it cannot tell whether a step took effect, it stops and asks a person.
* Each agent only sees and can use the tools it is allowed.
* It trims old context so the model does not overflow.

In the 200-run crash benchmark, restarting duplicated actions in 145 runs and resuming duplicated none.

## Input and output

**In:** an agent definition (its instructions and the tools it is allowed to use), a task in plain language, and a model.
Over HTTP that is `POST /runs {"agent": "writer", "task": "...", "task_kind": "..."}`.

**Out:** a run record. It holds the status (`done`, `failed`, or `needs_review`), the final answer, and every step with
its tool, arguments, result, state and retry count. The tools' real effects (files written, lines appended) happen
exactly once, even if the process was killed partway. Finished runs can also be distilled into skill files that the
next run of the same kind reads.

If you remove the model from an agent's architecture diagram, everything left over is the harness. Here
that is four things:

| Piece | What it owns | Where |
|---|---|---|
| **Loop** | Asks the model for one action at a time, enforces `max_steps`, records every step | `harness/runtime.py` |
| **Sandbox** | Per-agent tool allowlist, argument validation, path jail, call budgets, timeouts | `harness/sandbox.py` |
| **Run log** | SQLite record of every step, written before and after each tool call; resume after a crash | `harness/store.py` |
| **Memory** | Context budget with compaction, plus nightly distillation of past runs into reusable skills | `harness/context.py`, `harness/skills.py` |

The model plugs in through one method, `next_action(context, tools)`. `AnthropicModel` drives Claude with
tool use; `ScriptedModel` is a deterministic stand-in used by the tests and the benchmark.

## Crash recovery, step by step

Every tool call moves through four states in the log:

```
started ──► dispatched ──► done | error
   │             │
   │             └─ crash here: the tool may or may not have taken effect
   └─ crash here: the tool never ran
```

On `resume`:

* `done` and `error` steps are **replayed from the log**. Their results go back into the context and the tool is not called again.
* A `started` step **never ran**, so it is run now.
* A `dispatched` step is uncertain. If the tool is marked `idempotent` (overwriting a file, an upsert), it is run again. If it is not
  (appending a line, sending a message, charging a card), the run stops with status `needs_review` and says which step to check.
  The harness refuses to guess.

The model never sees the result of an interrupted step, because that result does not exist.

## Measured: restart vs resume

`bench/crash_bench.py` runs 200 eight-step tasks that mix idempotent file writes with non-idempotent audit-log
appends. It kills each run at a random step in one of the three windows above, then recovers it two ways with the
same seed. The model is scripted, so both strategies make identical decisions; only recovery differs.

| After a crash | Restart from scratch | Harness resume |
|---|---|---|
| Runs that ended with a duplicated side effect (extra audit line) | 145 / 200 | **0 / 200** |
| Side-effecting tool calls executed a second time | 748 | 39 (all idempotent overwrites) |
| Runs correct with no human involved | 55 / 200 | **143 / 200** |
| Runs stopped for human review | 0 / 200 | 57 / 200 |

How to read it:

* Restarting "works" in the sense that the task finishes, but in almost three of four runs it silently corrupts the audit log.
* Resume never duplicated a side effect. The price is that 57 runs stop and ask a person to check one step, because a
  non-idempotent tool was in flight when the process died. Some of those tools had not actually taken effect yet,
  but the log cannot know that, and a person can check in seconds.
* The way to shrink that 57 is to make more tools idempotent, for example appends that carry a dedupe key. That is the next
  thing on the list.
* This is a synthetic benchmark on local file tools. It measures the recovery logic, not any particular production system.

CI reruns the benchmark on every push and fails if resume ever produces a duplicated side effect.

Reproduce: `python -m bench.crash_bench --trials 200 --seed 1`

## Running the triage agent inside the harness

[pipeline-triage-graph](https://github.com/dhruvdubey3123/pipeline-triage-graph) decides *what* to fix and waits for a person to
approve it. `triage_bridge/` makes this harness the thing that *applies* the approved fix, through triage's own `Executor`
extension point. Both halves keep their state in SQLite (LangGraph's `SqliteSaver` for the graph, the run log for the harness),
so the process can be killed at any moment and a new process carries on.

![Triage inside the harness: 0 of 75 approved fixes duplicated an action after a crash, versus 61 of 75 when restarted](docs/triage_in_harness.png)

* **Only the approved plan can run.** By default the fixer's "model" is the plan itself, replayed call for call. Hand the job to
  a real model instead and the approval is still enforced, by the sandbox rather than by the prompt:
  [see below](#a-model-applies-the-fix-and-cannot-leave-the-approval).
* **Every action declares whether it is safe to repeat.** Patching a procedure (`CREATE OR REPLACE`) and rebuilding a table are;
  re-running a load, commenting on the ticket and writing the audit line are not.
* **A real `kill -9`.** `python -m triage_bridge.demo` runs each step as a separate process and kills one with `os._exit(137)`
  right after it re-ran a job. The new process stops for review, a person confirms the job ran, and the fix finishes with every
  action exactly once. Restarting the plan from the top re-runs the job a second time. Output: [triage_bridge/demo_output.txt](triage_bridge/demo_output.txt).

### Measured across the triage eval set

`python -m triage_bridge.bench --seed 1` takes all 100 labelled incidents. Triage sends 75 to the human gate and escalates 25, matching
the labels 100/100. Each of the 75 approved fixes is killed at a random step, in one of three windows (before the tool runs, after it
is dispatched but before it runs, after it ran but before the log recorded it), then recovered two ways.

| After the crash | Restart the approved plan from the top | Harness resume |
|---|---|---|
| Fixes that duplicated an action (extra load run, comment or audit line) | 61 / 75 | **0 / 75** |
| Duplicated actions in total | 95 | **0** |
| Fixes where every action happened exactly once | 14 / 75 | **75 / 75** |
| Finished without asking anyone | 75 / 75 | 45 / 75 |
| Paused for one human check | 0 | 30 / 75 |

The 30 pauses are the price: in those runs a non-repeatable action was in flight when the process died, so the harness asks instead
of guessing. CI runs this benchmark and the kill demo on every push and fails if resume ever duplicates or drops an action.
[The next section](#tuning-itself-against-its-own-benchmark) is a loop that searched for a safe way to cut those 30 pauses, and got
them to 0 without duplicating or dropping anything.

Limits: the warehouse is a SQLite stand-in, the reviewer's approval is automatic in the benchmark, and diagnosis uses triage's
rule-based path.

```bash
pip install -e ".[triage]"
python -m triage_bridge.demo
python -m triage_bridge.bench --seed 1
```

## A model applies the fix, and cannot leave the approval

Replaying an approved plan is safe and decides nothing. The moment a real model applies the fix instead, it starts choosing: the
order of the work, the wording of the ticket comment, what to do about an error, when to stop. That is the useful part. The risk
is that what a person approved and what happens drift apart, and no amount of prompting makes that a guarantee.

So the approved plan becomes an **envelope** rather than a script. `harness/envelope.py` turns the plan into a list of approved
actions, each with the fields that decide what it does to the world and a count of how many times it may happen:

```
patch_procedure(node='sp_fill_rate'); rerun_job(node='sp_fill_rate'); refresh_table(node='fct_workforce');
refresh_table(node='rpt_workforce'); comment_ticket(case_id='SYN-007'); append_audit()
```

* **The target is fixed, the prose is not.** `comment_ticket(case_id='SYN-007')` approves one comment on that case; the body is
  the model's to write. A comment on a different case is a different action and is refused.
* **Enforced in the sandbox, before dispatch.** A call outside the envelope never reaches the tool. It is written to the run log
  as a refusal with the reason, so an operator reading the log afterwards can tell an overreach from a failure.
* **Spent budgets survive a crash.** What is left of the approval is rebuilt from the log, so a second process cannot re-spend
  an action the first one already used. A refusal and a give-up both mean nothing happened and cost nothing; an unexplained
  failure is charged, because the conservative reading of an unknown effect is that it happened.

`python -m triage_bridge.live_fix` runs three checks against a local Ollama: a clean apply, a kill right after the job rerun
followed by a resume in a new process, and an untrusted note that arrived with the incident asking for two actions nobody
approved. CI runs the same script against a stub server that takes the note's advice, and asserts that the extra load run and
the second comment are both refused while the approved six still happen exactly once.

```bash
ollama pull qwen3:8b
pip install -e ".[triage,local]"
python -m triage_bridge.live_fix
```

This bounds what a model can do, not whether it is right. Its diagnosis can still be wrong, which is why the plan sits behind a
human gate and the envelope sits behind that.

## Two agents, one person, one log

With the fixer in place, the other half of the job is still deterministic: the graph finds the root cause by walking the pipeline
structure. `triage_bridge/orchestrate.py` runs the whole incident as two model-driven agents instead, orchestrated by the harness
rather than by a framework:

```
diagnose (model, read-only tools)  ->  review  ->  a person approves  ->  apply (model, inside the approved envelope)
```

The two agents are deliberately unequal.

* **The diagnoser has judgment and no power.** Its sandbox holds five read-only tools: the pipeline and its edges, the reported
  status of a node, everything upstream of a node, similar resolved cases, and `submit_diagnosis`. There is no tool that changes
  anything, so a wrong answer stays a wrong answer instead of becoming a wrong action.
* **What it submits is checked before a person sees it.** An unknown category, a node that is not in the pipeline, or a table
  nobody may patch directly come back as an error the model can read and correct. The plan itself is not up for negotiation:
  once the target is known, what else to re-run is pipeline structure, so it is derived rather than proposed.
* **The fixer has power and no judgment.** It writes to the warehouse, and only the approved actions, as above.
* **The diagnosis is read back out of the run log, not held in memory.** Kill the process during the diagnosis and the next one
  picks it up; kill it during the fix and the next one finishes without repeating an action. Both agents are runs in one SQLite
  log, so `trace()` prints the whole incident as one sequence.

```
[diagnose] done
  0 pipeline({}) -> done
  1 rerun_job({"node": "sp_fill_rate"}) -> error [denied: tool 'rerun_job' is not allowed for this agent]
  2 upstream({"node": "rpt_workforce"}) -> done
  3 precedents({"query": "invalid identifier after an upstream column rename"}) -> done
  4 submit_diagnosis({"category": "schema-drift", ...}) -> error [ValueError: 'schema-drift' is not a category...]
  5 submit_diagnosis({"category": "schema_drift", "cause": "load_cases", "fix_target": "sp_fill_rate"}) -> done
[apply] done
  0 patch_procedure({"node": "sp_fill_rate", "change": "schema_drift"}) -> done
  1 rerun_job({"node": "load_cases"}) -> error [denied: ... is not in the approved plan]
  2 rerun_job({"node": "sp_fill_rate"}) -> done
  ...
```

### Is the model any good at it?

The 100 incidents carry labels, so this is measurable rather than a matter of opinion. The deterministic tracer sets the bar:

| On all 100 labelled incidents | Rule-based tracer |
|---|---|
| Root cause identified correctly | 100 / 100 |
| Category identified correctly | 93 / 100 |
| Wall clock | 0.8 s |

`python -m triage_bridge.diag_bench --limit 20` scores a model against the same labels and prints both columns side by side,
including which cases it missed. Nothing is applied during that benchmark, so a wrong answer costs a wrong row and nothing else.
Fill in the model column from your own run; a small local model is not expected to beat a tracer written for this pipeline, and
the useful question is how close it gets and where it fails.

```bash
python -m triage_bridge.diag_bench --limit 20   # the model diagnoses, scored against the labels
python -m triage_bridge.live_fix                # 4 checks: apply, crash, untrusted input, both agents
```

CI runs the same two-agent path against the stub server, including a kill during the diagnosis and a kill during the fix.

## Tuning itself against its own benchmark

Those 30 pauses were tuned by hand. `evolve/` makes that a loop instead: it proposes a change to the
recovery policy, measures it on the same crashed fixes, and keeps it only if the objective went up **and**
nothing was duplicated or dropped.

```
shipped config ──► propose ──► score ──► passed the gate and better? ──► new best
                      ▲                            │ no
                      └──────── next candidate ◄───┘
```

The interesting part is not the search, it is what stops it. A loop that can change anything can change the
thing that judges it, so:

* **It can only move a typed record**, never code: a recovery strategy per non-repeatable tool, the retry
  limit, the gate's confidence threshold. Anything else is rejected before it is scored.
* **Frozen:** the tool allowlists, which tools are marked safe to repeat, the human gate, the benchmark and
  its labels, and the gate's own checks.
* **Nothing merges itself.** A run writes a ledger and a person reads it.

### The benchmark alone is not a good enough judge

None of the 100 incidents happens to contain a plan that runs the same job twice, so a policy that is wrong
in exactly that case would score perfectly. Four hand-written counterexamples cover it, each built to break a
specific shortcut:

| Policy for a non-repeatable step left in flight | same job twice, killed before the 2nd run | same job twice, killed after it | one-off action, killed before | after |
|---|---|---|---|---|
| `pause` (shipped) | paused | paused | paused | paused |
| `unique_payload` | paused | paused | **clean** | **clean** |
| `assume_done` | **dropped** | clean | paused | paused |
| `retry` | clean | **duplicated** | paused | paused |

`unique_payload` asks the warehouse whether this exact call is recorded there, but only when no earlier step
in the same run made an identical call: without that guard it would read the earlier call's trace as evidence
for this one and silently skip a job. `assume_done` and `retry` exist so the search space contains wrong
answers; a gate that cannot reject anything is not a gate. A test asserts both are still caught, because if
they stop being caught every later result is worthless.

### What one run found

![The loop cut human interruptions after a crash from 30 to 0, and the gate rejected four candidates including two that scored a perfect 75/75](docs/evolve.png)

`python -m evolve --generations 9 --seed 1`, 10 configurations on the same 75 crashed fixes:

| Configuration | Finished without a person | Duplicated | Dropped | Verdict |
|---|---|---|---|---|
| all tools pause (shipped) | 45 / 75 | 0 | 0 | baseline |
| `rerun_job` verified | 57 / 75 | 0 | 0 | better |
| `+ comment_ticket` | 67 / 75 | 0 | 0 | better |
| **`+ append_audit`** | **75 / 75** | **0** | **0** | **kept** |
| `rerun_job=assume_done` | not scored | | | rejected by a counterexample |
| `rerun_job=retry` | not scored | | | rejected by a counterexample |
| `comment_ticket=assume_done` | 75 / 75 | 0 | **5** | rejected: dropped 5 actions |
| `comment_ticket=retry` | 75 / 75 | **5** | 0 | rejected: duplicated 5 actions |

**Human interruptions after a crash went from 30 to 0 with nothing duplicated and nothing dropped.** The last
two rows are the reason the gate exists: both hit a perfect score on the thing being optimised, by quietly
skipping or repeating five real actions. The objective alone would have accepted either.

Honest about what that 100% means: no plan in these 100 incidents runs the same action twice, so every
interrupted step was uniquely identifiable and the warehouse could be asked about it. The counterexamples
show the policy going straight back to pausing the moment a plan does repeat one. The verifier is a query
against the system of record, the same check an on-call engineer makes by hand, not an oracle.

```bash
python -m evolve --probes                  # just the counterexamples
python -m evolve --generations 9           # the loop, ledger in evolve/results/
python -m evolve --proposer model          # the proposer is a local model (needs Ollama)
```

With `--proposer model` the proposer is itself an agent inside this harness, holding one tool and nothing
else in its allowlist, so its suggestion comes back as a `propose_change` call and faces the same gate as
any other candidate. CI runs the counterexamples, asserts they still catch both unsafe strategies, and runs
a short search that fails if the winner ever duplicates or drops an action.

## Tested with a real open-weight model

`examples/real_model.py` against **Qwen3 8B** served by Ollama on a laptop CPU: **4 of 4 checks passed** in 201 s.

![Real-model results: 4 of 4 checks passed](docs/real_model.png)

* A clean run, a crash after a safe write, and a crash after a non-idempotent append all ended with the audit line written exactly once. Restarting from scratch after the same crash wrote it twice.
* The model's proposed SQL fix was wrong (it dropped a column instead of fixing the reference). The harness controls how an agent acts, not whether its idea is right, which is why risky actions sit behind a human gate.
* Full output: [examples/results/qwen3-8b.md](examples/results/qwen3-8b.md).

## Quick start

```bash
pip install -e ".[dev]"
pytest -q                          # 32 tests, 68 with the [triage] extra
python examples/quickstart.py      # run, crash at step 2, resume from the log
```

```
crashed at step 2
0 write_file done
1 append_line done
2 read_file done
3 append_line done
status: done | replayed from log: 2
audit.log: ['notes created', 'notes checked']
```

With a free open-weight model, no API key (any OpenAI-compatible server works: Ollama, llama.cpp, vLLM, LM Studio):

```bash
ollama pull qwen3:8b
pip install -e ".[local]"
python examples/real_model.py      # clean run, two crash-and-resume runs, and a sandbox check, against the real model
HARNESS_BASE_URL=http://localhost:11434/v1 uvicorn harness.api:app
```

With Claude:

```bash
pip install -e ".[llm]"
export ANTHROPIC_API_KEY=...       # HARNESS_MODEL to override the model id
uvicorn harness.api:app            # POST /runs, GET /runs/{id}, POST /runs/{id}/resume, POST /skills/distill
```

## Sandbox

```python
reader = AgentSpec("reader", "Answer from files only.", Policy({"read_file", "list_files"}))
```

* The model is only shown tools in its policy. A reader agent cannot discover that `write_file` exists.
* Arguments are validated against each tool's Pydantic schema before anything runs.
* Paths containing `..`, `.git`, `.env` or `.ssh` are refused, and file tools are jailed to the workspace root as a second layer.
* Each tool has a per-run call budget and a timeout. Timeouts count as retryable.
* Denials are returned to the model as an error result, so it can change course, and are never retried.

### Least-privilege MCP

The same policies can be served over MCP, one server per policy, so an MCP client gets exactly the tools that agent is
allowed and nothing else:

```bash
python -m harness.mcp_bridge --workspace ./ws --allow read_file,list_files
```

## Retries

Tools raise `RetryableToolError` for transient failures such as rate limits or flaky networks. The harness retries with
exponential backoff up to `max_retries`, records the attempt count on the step, and then hands the final error to the
model. Bugs and denials are not retried.

## Context budget

The harness, not the model, decides what fits. When the estimated context exceeds the budget, the oldest tool results
are compacted to a one-line stub, while the system prompt, the task and the most recent steps stay intact. The
estimate is characters divided by four, which is crude on purpose; swap in a tokenizer if you need precision.

## Skills from past runs

`distill(store, "skills/")` reads finished runs, groups them by `task_kind`, and when the same tool sequence has succeeded
at least `min_support` times, writes it to `skills/<kind>.md` plus an `index.json`. The next run of that kind gets the
procedure in its system prompt. The skill files are plain markdown, so you can read, edit or delete what was learned.
`k8s/deployment.yaml` includes a nightly CronJob that triggers this.

## Deploy

Slim non-root `Dockerfile`. `k8s/deployment.yaml` runs a single replica, because the run log is SQLite on a persistent
volume. Swap `RunStore` for Postgres to scale out; nothing else changes. It also includes probes, two volume claims, a
Service and the distillation CronJob.

## Status

Built and tested, including one end-to-end run with a real open-weight model (Qwen3 8B, above). Not yet
battle-tested at scale. The triage agent runs inside it, a model can diagnose an incident with read-only
tools and then apply the approved fix without being able to leave the approval, and the harness tunes its
own recovery policy against its benchmark behind a gate that rejected 4 of 10 candidates (all above).
A Postgres store and distributed tracing are next.

MIT licensed.
