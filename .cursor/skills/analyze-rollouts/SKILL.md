---
name: analyze-rollouts
description: Analyze rollout result directories and produce per-task evaluation tables with outcome, step count, task id, and client-error status. Use when the user asks for evaluation statistics, fail/success tables over runs, per-task rollout summaries, or a subset excluding client-error runs.
---

# Analyze Rollouts

Use this skill to summarize a rollout results directory into two markdown tables:

- all runs
- runs without client error

## Definitions

- A run is any directory under the target results root that contains both `result.txt` and `traj.jsonl`.
- Sort runs by the first parsed `action_timestamp` in `traj.jsonl` ascending. If timestamps tie or are missing, fall back to `task_id`.
- `outcome` is `success` when `result.txt` parses to `1.0`, `fail` when it parses to `0` or `0.0`, otherwise `score=<raw text>`.
- `steps` is the number of parsed JSON lines in `traj.jsonl`.
- `client_error` is `true` when any trajectory row has `action == "DONE"` and `response` contains `client error` case-insensitively.

## Workflow

1. Confirm the target results directory.
2. Activate the local environment before terminal commands:

```bash
source .venv/bin/activate
```

3. Run the helper script:

```bash
python .cursor/skills/analyze-rollouts/scripts/analyze_rollouts.py GUI-Docker-Env/results_thunderbird_all_ui_tars_existing_emulator
```

4. Return the two markdown tables from the script output:
   - `All Runs`
   - `Runs Without Client Error`

## Output Table Schema

Both tables use these columns:

- `#`
- `outcome`
- `steps`
- `task_id`
- `client_error`

For the second table, only include rows where `client_error` is `false`.
