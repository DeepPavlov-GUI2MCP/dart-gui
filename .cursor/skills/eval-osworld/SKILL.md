---
name: eval-osworld
description: Run OSWorld GUI evaluations with UI-TARS over dart-rollouter vLLM, including parallel multi-emulator runs. Use when evaluating OSWorld tasks, running uitars_run/run_uitars.py, checking running emulator quota, choosing emulator/token concurrency, smoke-testing a model endpoint, or checking desktop server + vLLM health before a full eval.
disable-model-invocation: true
---

# Eval OSWorld

Run UI-TARS-1.5 evaluations against OSWorld tasks on the local desktop server, with inference on **dart-rollouter** vLLM over SSH port-forward.

## Required: check quota, then ask emulator allocation

**Step 1 — always check running emulators first** (before asking how many to run):

```bash
bash .cursor/skills/eval-osworld/scripts/check_emulator_quota.sh
```

Report the output to the user. Include:

- How many emulators are running (list ids if any)
- Token limit, in-use count, and **free slots**
- If any emulators are running: warn that if the requested parallel count **exceeds free slots**, those emulators **must be stopped** before the eval can start

Example notice when emulators are active:

> 2 emulators are running (0 free slots). If you want N parallel workers and N is greater than free slots, I will need to stop the running emulators first. Should I proceed?

**Step 2 — ask the user** (only after Step 1):

> How many emulators (token quota) should I allocate for this run?

Do **not** assume `--max-workers` or token limit. Wait for an explicit number (typically `1` or `2` on this machine).

**Step 3 — after the user answers:**

1. If `requested > free_slots` and emulators are running: **confirm**, then stop them:

```bash
curl -s -X POST http://127.0.0.1:50003/stop_all_emulators
bash .cursor/skills/eval-osworld/scripts/check_emulator_quota.sh   # verify free slots
```

2. Set `--max-workers` to the requested number (parallel task count = parallel emulators).
3. Set the desktop-server token limit to match (see **Parallel execution** below).
4. Run preflight with `EMULATOR_COUNT=<n>` so quota is validated before tasks start.

If the user did not specify concurrency, ask — do not default to `8`.

## Canonical model names

Use these spellings in docs, configs, and `--model` when the id is registered on vLLM:

| Purpose | Name |
|---------|------|
| HuggingFace / rollouter ckpt | `ByteDance-Seed/UI-TARS-1.5-7B` |
| CLI alias (default in `run_uitars.py`) | `ui_tars_1.5` |

**Always resolve the live vLLM model id before eval.** vLLM often exposes a local snapshot path (not the HF name). Query `/v1/models` and use the returned `id` for sanity checks; `UITARSAgent` auto-resolves aliases at runtime.

## Prerequisites

1. **SSH tunnel** to dart-rollouter — must stay up for the **entire** eval (parallel runs amplify impact of drops):

```bash
ssh -N dart-rollouter
```

Or a persistent tmux session:

```bash
tmux new-session -d -s dart-rollouter-tunnel \
  "ssh -N -o ExitOnForwardFailure=yes dart-rollouter"
```

Expected local forwards (from `~/.ssh/config`):

| Local | Remote | Service |
|-------|--------|---------|
| `127.0.0.1:8010` | vLLM | OpenAI-compatible inference |
| `127.0.0.1:15961` | model service | Rollouter status API |

2. **Desktop server** on port 50003 (see `AGENTS.md`).

3. **Venv** at repo root:

```bash
source .venv/bin/activate
```

4. **Env** — copy defaults if missing:

```bash
cp GUI-Docker-Env/.env-default GUI-Docker-Env/.env
```

Ensure `OPENAI_BASE_URL=http://127.0.0.1:8010` is set (do not leave it blank in `.env`; empty overrides the default).

Optional override when vLLM id is known:

```bash
export OPENAI_MODEL='<resolved-id-from-v1-models>'
```

## Parallel execution

`--max-workers` = number of OSWorld tasks (and emulators) running **at the same time**. Each task gets its own `DesktopEnv` → own QEMU container.

| Setting | Meaning |
|---------|---------|
| `--max-workers N` | Up to N tasks/emulators in parallel |
| Token limit (`dart`) | Max emulators the desktop server allows for that token |
| Config default | `GUI-Docker-Env/configs/config.yaml` → `tokens.dart: 2` |

**Rules:**

- `--max-workers` must be **≤** token limit, or emulator starts fail with HTTP 429 quota exceeded.
- Always call `/set_token_limit` before eval so quota matches the user's requested allocation.
- Keep SSH tunnel alive; connection drops mid-run cause `APIConnectionError` and failed tasks.
- Parallel runs share one vLLM instance — throughput may not scale linearly.

### Set token limit (required before parallel eval)

```bash
# Replace 2 with the number the user requested
curl -s -X POST http://127.0.0.1:50003/set_token_limit \
  -H "Content-Type: application/json" \
  -d '{"token":"dart","limit":2}'
```

Verify quota:

```bash
curl -s http://127.0.0.1:50003/status | python3 -m json.tool
# tokens[].limit, current, available
```

During a run, expect `current == max-workers` until tasks finish. After completion, `current` should return to 0.

## Preflight (required before full eval)

Run from repo root. Checks desktop server, model service, vLLM health, lists models, resolves UI-TARS id, sanity `chat/completions`, and **validates emulator quota** when `EMULATOR_COUNT` is set.

```bash
EMULATOR_COUNT=2 bash .cursor/skills/eval-osworld/scripts/preflight.sh
```

Single emulator (default check):

```bash
EMULATOR_COUNT=1 bash .cursor/skills/eval-osworld/scripts/preflight.sh
```

Custom model hint:

```bash
REQUESTED_MODEL='ByteDance-Seed/UI-TARS-1.5-7B' EMULATOR_COUNT=2 \
  bash .cursor/skills/eval-osworld/scripts/preflight.sh
```

Do **not** start a full eval if preflight exits non-zero.

## Run evaluation

All commands use `GUI-Docker-Env/` as cwd and **`PYTHONPATH=.`** (required).

Replace `N` below with the emulator count the user confirmed.

### Single-task smoke (1 emulator)

Task file: `evaluation_examples/test_writer_first.json` (one task).

```bash
source .venv/bin/activate
cd GUI-Docker-Env

curl -s -X POST http://127.0.0.1:50003/set_token_limit \
  -H "Content-Type: application/json" \
  -d '{"token":"dart","limit":1}'

EMULATOR_COUNT=1 bash ../.cursor/skills/eval-osworld/scripts/preflight.sh

PYTHONPATH=. python uitars_run/run_uitars.py \
  --domain libreoffice_writer \
  --test-all-meta-path evaluation_examples/test_writer_first.json \
  --max-workers 1 \
  --max-steps 15 \
  --result-dir results_uitars15_writer_first \
  --overwrite
```

### Parallel smoke (2 writer tasks, 2 emulators)

Task file: `evaluation_examples/test_writer_two.json` (first two `libreoffice_writer` tasks).

```bash
source .venv/bin/activate
cd GUI-Docker-Env

curl -s -X POST http://127.0.0.1:50003/set_token_limit \
  -H "Content-Type: application/json" \
  -d '{"token":"dart","limit":2}'

EMULATOR_COUNT=2 bash ../.cursor/skills/eval-osworld/scripts/preflight.sh

PYTHONPATH=. python uitars_run/run_uitars.py \
  --domain libreoffice_writer \
  --test-all-meta-path evaluation_examples/test_writer_two.json \
  --max-workers 2 \
  --max-steps 15 \
  --result-dir results_uitars15_writer_parallel_$(date +%Y%m%d_%H%M%S) \
  --overwrite
```

Confirm both emulators started:

```bash
curl -s http://127.0.0.1:50003/status | python3 -c \
  "import json,sys; s=json.load(sys.stdin); t=s['tokens'][0]; print(f\"emulators={s['total_emulators']} current={t['current']} limit={t['limit']}\")"
```

### Domain subset (N emulators)

```bash
PYTHONPATH=. python uitars_run/run_uitars.py \
  --domain libreoffice_writer \
  --test-all-meta-path evaluation_examples/test_all.json \
  --max-workers N \
  --max-steps 15 \
  --result-dir results_uitars15_writer_$(date +%Y%m%d_%H%M%S)
```

Resume skips finished tasks unless `--overwrite` is set.

### Full benchmark (nogdrive split)

Ask the user for emulator count first. Do not use `--max-workers 8` unless token limit is raised in config and the user explicitly requests it.

```bash
PYTHONPATH=. python uitars_run/run_uitars.py \
  --test-all-meta-path evaluation_examples/test_nogdrive.json \
  --max-workers N \
  --max-steps 15 \
  --result-dir results_uitars15_nogdrive_$(date +%Y%m%d_%H%M%S)
```

## Common flags

| Flag | Default | Notes |
|------|---------|-------|
| `--model` | `ui_tars_1.5` | Alias; agent resolves via `/v1/models` |
| `--model-type` | `qwen25vl` | Keep for UI-TARS-1.5 |
| `--max-workers` | `8` | **Override** — set to user-requested emulator count |
| `--max-steps` | `15` | OSWorld step budget per task |
| `--result-dir` | `./results_chenrui` | Use timestamped dirs for new runs |
| `--overwrite` | off | Re-run and replace existing results |
| `--base-url` | `http://127.0.0.1:50003` | Desktop server |
| `--token` | `dart` | Desktop server auth |

## Results layout

```
{result_dir}/pyautogui/screenshot/{model}/{domain}/{task_id}/
  result.txt      # 1 = pass, 0 = fail
  traj.jsonl      # step-by-step trajectory
```

Runner `success=2 failed=0` means both tasks **completed without crashing**, not that both passed (`result.txt` may be 0).

Quick pass-rate check: use **analyze-rollouts** skill.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: mm_agents` | Add `PYTHONPATH=.` |
| `The model ui_tars_1.5 does not exist` | Run preflight; set `OPENAI_MODEL` to resolved id |
| vLLM 404 / connection refused | Restore SSH tunnel; verify `curl http://127.0.0.1:8010/health` |
| `APIConnectionError` mid-run | SSH tunnel dropped — restart tunnel, re-run failed tasks |
| `Token quota exceeded` (429) | Lower `--max-workers` or raise limit via `/set_token_limit` |
| Only 1 emulator despite `--max-workers 2` | Check `/status` — quota may be 1; call `/set_token_limit` |
| Stuck emulators after crash | `curl -X POST http://127.0.0.1:50003/stop_all_emulators` |
| Need free slots before parallel eval | Run `check_emulator_quota.sh`; stop running emulators if user confirms |

## Utility scripts

| Script | When |
|--------|------|
| `scripts/check_emulator_quota.sh` | **First** — before asking user for parallel count |
| `scripts/preflight.sh` | After user confirms count — health + model sanity + quota validation |

## Related skills

- **dart-rollouter** — start model service on GPU VM
- **debug-osworld** — sequential rerun until first client error
- **analyze-rollouts** — per-task outcome tables from result dirs
