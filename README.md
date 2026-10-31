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

## Quick start

```bash
pip install -e ".[dev]"
pytest -q                          # 18 tests
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

With a real model:

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

Built and tested. Not yet battle-tested against a live model at scale. Real-model runs, a Postgres store and
OpenTelemetry spans are next.

MIT licensed.
