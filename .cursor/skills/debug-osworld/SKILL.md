---
name: debug-osworld
description: Debug OSWorld or GUI-Docker-Env evaluation runs by launching tasks sequentially until the first client error, while saving per-task timing, step counts, start/end timestamps, and run logs. Use when rerunning a subset of OSWorld tasks, reproducing client-error failures, or collecting task-by-task timing stats.
disable-model-invocation: true
---

# Debug OSWorld

Use this skill when you need a deterministic sequential rerun that stops at the first client-error task and saves structured task metrics.

## Workflow

1. Activate the project environment:

```bash
source .venv/bin/activate
```

2. Run the helper script from the repo root. Pick a fresh results directory so existing runs are untouched.

Thunderbird example:

```bash
python .cursor/skills/debug-osworld/scripts/run_until_client_error.py \
  --test-all-meta-path validation/evaluation_examples/test_thunderbird_all.json \
  --domain thunderbird \
  --result-dir GUI-Docker-Env/results_thunderbird_debug_$(date +%Y%m%d_%H%M%S) \
  --model ByteDance-Seed/UI-TARS-1.5-7B \
  --model-type qwen25vl \
  --max-steps 30
```

All-task example:

```bash
python .cursor/skills/debug-osworld/scripts/run_until_client_error.py \
  --test-all-meta-path GUI-Docker-Env/evaluation_examples/test_all.json \
  --result-dir GUI-Docker-Env/results_debug_client_error_$(date +%Y%m%d_%H%M%S)
```

3. Watch the console for one of these markers:

- `FIRST_CLIENT_ERROR <domain>/<task_id> <task_dir>`
- `NO_CLIENT_ERROR_FOUND`

4. If a client error is found, inspect the remote model or server logs immediately using the task timestamp and task id.

## What Gets Saved

The normal OSWorld artifacts still go into the chosen `--result-dir`.

The helper also creates `--result-dir/_debug_osworld/` with:

- `session.log`: full console stream for the sequential debug run
- `task_stats.jsonl`: one JSON object per finished task
- `task_stats.csv`: flat table with the same per-task stats
- `summary.json`: current run summary, including finished task count and whether a client error stopped the loop

Each task row includes:

- `domain`
- `task_id`
- `status`
- `outcome`
- `raw_result`
- `steps`
- `client_error`
- `started_at`
- `ended_at`
- `elapsed_sec`
- `example_dir`

`summary.json` also records the standard `GUI-Docker-Env/logs/normal-*`, `debug-*`, and `sdebug-*` files created by `run_uitars.py`.

## Notes

- The helper uses the same `run_one_example(...)` path as `GUI-Docker-Env/uitars_run/run_uitars.py`, but runs tasks one by one.
- Client error is detected from `traj.jsonl`: a row with `action == "DONE"` whose `response` contains `client error` case-insensitively.
- Use `--task-limit` when you only want the first N tasks from the selected meta file.
