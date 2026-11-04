# Real-model run: Qwen3 8B on a laptop

`python examples/real_model.py`, run on a Windows laptop (CPU) with Ollama serving `qwen3:8b`.
4 of 4 checks passed in 201 s. Output below is copied from the terminal, trimmed only of pip noise.

## 1. Clean run: passed
```
status: done
steps:  0 read_file -> done | 1 write_file -> done | 2 append_line -> done
audit.log: Fix plan written for AGS-4471
fix_plan.md:
  Root Cause: The SQL query references the obsolete column name 'CUST_ID' which was renamed to
  'CUSTOMER_ID' in the CRM.CUSTOMERS table.
  Fix: ALTER TABLE CRM.DAILY_REVENUE DROP COLUMN CUST_ID;
```
The harness did its job: three tool calls, each once, run marked done. **The model's fix is wrong.** It
diagnosed the cause correctly, then proposed dropping a column instead of renaming the reference in
`sp_load_daily_revenue`. The harness guarantees *how* an agent acts (once, in its sandbox, on the record),
not *whether the idea is good*. That is what a human approval gate is for, as in
[pipeline-triage-graph](https://github.com/dhruvdubey3123/pipeline-triage-graph).

## 2. Killed right after `write_file` (safe to repeat), then resumed: passed
```
crashed: True     status after resume: done     replayed_from_log: 1
steps:  0 read_file -> done | 1 write_file -> done | 2 append_line -> done
audit.log: Fix plan written for AGS-4471        (one line)
```

## 3. Killed right after `append_line` (not safe to repeat): passed
```
audit.log at crash:                 1 line
resume:                             needs_review   (step 2 append_line was dispatched)
audit.log after resume:             1 line
audit.log after restart from zero:  2 lines        <- the duplicate the harness prevents
```

## 4. Read-only agent asked to write: passed, with a caveat
```
status: done     tool calls: none     files: incident.md only
answer: "I cannot write to files directly. ... Would you like me to read the file instead?"
```
The write tool was never shown to this agent, so the model declined on its own. The file was not
created, but the sandbox's refusal path (a denied call) was not exercised by the real model this time;
that path is covered by the unit tests.
