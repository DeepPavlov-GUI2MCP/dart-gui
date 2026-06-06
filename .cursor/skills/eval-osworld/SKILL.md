name: eval-osworld
description: Run OSWorld GUI evaluations against dart-rollouter vLLM in a dedicated tmux session, choosing the correct runner for the live model on the remote host. Use when evaluating OSWorld tasks, checking running emulator quota, choosing emulator/token concurrency, smoke-testing a model endpoint, or checking desktop server + vLLM health before a full eval.
disable-model-invocation: true
---

# Eval OSWorld

Run OSWorld evaluations against the local desktop server, with inference on **dart-rollouter** vLLM over SSH port-forward.

## Runner selection (required)

Do **not** assume UI-TARS. First inspect the live model id exposed by vLLM:

```bash
curl -s http://127.0.0.1:8010/v1/models | python3 -c "import sys,json; print([m['id'] for m in json.load(sys.stdin)['data']])"
```

Choose the eval entrypoint from the returned model id:

| Live model id contains | Runner | Agent/parser family |
|------------------------|--------|---------------------|
| `ui-tars`, `UI-TARS`, `ByteDance-Seed/UI-TARS-1.5-7B` | `uitars_run/run_uitars.py` | UI-TARS parser |
| `holo`, `Holo`, `holotron`, `Holotron`, `Hcompany/` | `holo_run/run_holo.py` | Holo structured JSON parser |

If unsure:

- prefer `holo_run/run_holo.py` for `Hcompany/Holotron-*` and `Hcompany/Holo*`
- prefer `uitars_run/run_uitars.py` only for actual UI-TARS models

Using the wrong runner causes parse failures even when the HTTP call succeeds.

## Tmux session (required)

**Default:** launch the selected runner (`uitars_run/run_uitars.py` or `holo_run/run_holo.py`) in a **dedicated** detached tmux session named `osworld-eval`. Do **not** run long evals in the agent’s current shell unless the user explicitly asks to (e.g. “run in foreground”, “no tmux”).

| Session | Purpose |
|---------|---------|
| `dart-rollouter-tunnel` | SSH port-forward only — **connect-dart-rollouter** skill |
| `osworld-eval` | Selected OSWorld eval runner only — this skill |

Do not combine tunnel and eval in one tmux session. Preflight, quota checks, and `/set_token_limit` run in the current shell; only the long-running eval goes in `osworld-eval`.

```bash
# After preflight — preferred
EVAL_CMD='source /path/to/dart-gui/.venv/bin/activate && cd /path/to/dart-gui/GUI-Docker-Env && PYTHONPATH=. python <selected-runner>.py ...' \
  bash .cursor/skills/eval-osworld/scripts/run_eval.sh start

bash .cursor/skills/eval-osworld/scripts/run_eval.sh status   # running? recent log tail
tmux attach -t osworld-eval                                   # live output; Ctrl+B then D to detach
bash .cursor/skills/eval-osworld/scripts/run_eval.sh stop     # kill eval session
```

**Opt-out:** If the user says to run in the foreground or without tmux, run `PYTHONPATH=. python <selected-runner>.py ...` directly and warn that closing the terminal will stop the eval.

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
5. Start the selected runner via `run_eval.sh start` in tmux `osworld-eval` (unless the user opted out of tmux).

If the user did not specify concurrency, ask — do not default to `8`.

## Canonical model names

Use these spellings in docs, configs, and `--model` / `--openai-model` when the id is registered on vLLM:

| Purpose | Name |
|---------|------|
| HuggingFace / rollouter ckpt | `ByteDance-Seed/UI-TARS-1.5-7B` |
| CLI alias (default in `run_uitars.py`) | `ui_tars_1.5` |
| Holo / Holotron family example | `Hcompany/Holotron-3-Nano` |

**Always resolve the live vLLM model id before eval.** vLLM often exposes a local snapshot path (not the HF name). Query `/v1/models` and use the returned `id` both for runner selection and for sanity checks.

## Prerequisites

1. **SSH tunnel** to dart-rollouter — must stay up for the **entire** eval. Use **connect-dart-rollouter** (separate tmux session `dart-rollouter-tunnel`):

```bash
.cursor/skills/connect-dart-rollouter/scripts/tunnel.sh start
.cursor/skills/connect-dart-rollouter/scripts/tunnel.sh check
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

Optional override when the vLLM id is known:

```bash
export OPENAI_MODEL='<resolved-id-from-v1-models>'
```

For Holo runs, also pass the resolved id via `--openai-model` when needed.

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

Setup (current shell): `set_token_limit`, `preflight.sh`. **Start eval** via `run_eval.sh` in tmux unless the user opted out.

All eval commands use `GUI-Docker-Env/` as cwd and **`PYTHONPATH=.`** (required). Use the repo’s absolute path in `EVAL_CMD` when starting tmux from arbitrary cwd.

Replace `N` below with the emulator count the user confirmed. Set `REPO` to the dart-gui root (e.g. `/home/pitchblack/dart-gui`).

### Single-task smoke (1 emulator, UI-TARS)

Task file: `evaluation_examples/test_writer_first.json` (one task).

```bash
REPO=/path/to/dart-gui
source "${REPO}/.venv/bin/activate"
cd "${REPO}/GUI-Docker-Env"

curl -s -X POST http://127.0.0.1:50003/set_token_limit \
  -H "Content-Type: application/json" \
  -d '{"token":"dart","limit":1}'

EMULATOR_COUNT=1 bash "${REPO}/.cursor/skills/eval-osworld/scripts/preflight.sh"

EVAL_CMD="source ${REPO}/.venv/bin/activate && cd ${REPO}/GUI-Docker-Env && PYTHONPATH=. python uitars_run/run_uitars.py \
  --domain libreoffice_writer \
  --test-all-meta-path evaluation_examples/test_writer_first.json \
  --max-workers 1 \
  --max-steps 15 \
  --result-dir results_uitars15_writer_first \
  --overwrite"
bash "${REPO}/.cursor/skills/eval-osworld/scripts/run_eval.sh" start
```

### Single-task smoke (1 emulator, Holo / Holotron)

```bash
REPO=/path/to/dart-gui
source "${REPO}/.venv/bin/activate"
cd "${REPO}/GUI-Docker-Env"

curl -s -X POST http://127.0.0.1:50003/set_token_limit \
  -H "Content-Type: application/json" \
  -d '{"token":"dart","limit":1}'

REQUESTED_MODEL='Hcompany/Holotron-3-Nano' EMULATOR_COUNT=1 \
  bash "${REPO}/.cursor/skills/eval-osworld/scripts/preflight.sh"

EVAL_CMD="source ${REPO}/.venv/bin/activate && cd ${REPO}/GUI-Docker-Env && PYTHONPATH=. python holo_run/run_holo.py \
  --domain libreoffice_writer \
  --test-all-meta-path evaluation_examples/test_writer_first.json \
  --max-workers 1 \
  --max-steps 15 \
  --model Hcompany/Holotron-3-Nano \
  --openai-model Hcompany/Holotron-3-Nano \
  --result-dir results_holo_writer_first \
  --overwrite"
bash "${REPO}/.cursor/skills/eval-osworld/scripts/run_eval.sh" start
```

### Parallel smoke (2 writer tasks, 2 emulators, UI-TARS)

Task file: `evaluation_examples/test_writer_two.json` (first two `libreoffice_writer` tasks).

```bash
REPO=/path/to/dart-gui
source "${REPO}/.venv/bin/activate"
cd "${REPO}/GUI-Docker-Env"

curl -s -X POST http://127.0.0.1:50003/set_token_limit \
  -H "Content-Type: application/json" \
  -d '{"token":"dart","limit":2}'

EMULATOR_COUNT=2 bash "${REPO}/.cursor/skills/eval-osworld/scripts/preflight.sh"

RESULT_DIR="results_uitars15_writer_parallel_$(date +%Y%m%d_%H%M%S)"
EVAL_CMD="source ${REPO}/.venv/bin/activate && cd ${REPO}/GUI-Docker-Env && PYTHONPATH=. python uitars_run/run_uitars.py \
  --domain libreoffice_writer \
  --test-all-meta-path evaluation_examples/test_writer_two.json \
  --max-workers 2 \
  --max-steps 15 \
  --result-dir ${RESULT_DIR} \
  --overwrite"
bash "${REPO}/.cursor/skills/eval-osworld/scripts/run_eval.sh" start
```

Confirm both emulators started (after a short wait):

```bash
curl -s http://127.0.0.1:50003/status | python3 -c \
  "import json,sys; s=json.load(sys.stdin); t=s['tokens'][0]; print(f\"emulators={s['total_emulators']} current={t['current']} limit={t['limit']}\")"
bash "${REPO}/.cursor/skills/eval-osworld/scripts/run_eval.sh" status
```

### Domain subset (N emulators, choose runner for live model)

```bash
EVAL_CMD="source ${REPO}/.venv/bin/activate && cd ${REPO}/GUI-Docker-Env && PYTHONPATH=. python <selected-runner>.py \
  --domain libreoffice_writer \
  --test-all-meta-path evaluation_examples/test_all.json \
  --max-workers N \
  --max-steps 15 \
  --result-dir results_osworld_writer_$(date +%Y%m%d_%H%M%S)"
bash "${REPO}/.cursor/skills/eval-osworld/scripts/run_eval.sh" start
```

Resume skips finished tasks unless `--overwrite` is set.

### Full benchmark (nogdrive split, choose runner for live model)

Ask the user for emulator count first. Do not use `--max-workers 8` unless token limit is raised in config and the user explicitly requests it.

```bash
EVAL_CMD="source ${REPO}/.venv/bin/activate && cd ${REPO}/GUI-Docker-Env && PYTHONPATH=. python <selected-runner>.py \
  --test-all-meta-path evaluation_examples/test_nogdrive.json \
  --max-workers N \
  --max-steps 15 \
  --result-dir results_osworld_nogdrive_$(date +%Y%m%d_%H%M%S)"
bash "${REPO}/.cursor/skills/eval-osworld/scripts/run_eval.sh" start
```

## Common flags

| Flag | Runner | Notes |
|------|--------|-------|
| `--model` | both | Use the resolved live model id when possible |
| `--openai-model` | Holo | Set to the resolved vLLM model id when Holo model name and API model id differ |
| `--model-type` | UI-TARS | Keep `qwen25vl` for UI-TARS-1.5 |
| `--max-workers` | both | **Override** — set to user-requested emulator count |
| `--max-steps` | both | OSWorld step budget per task |
| `--result-dir` | both | Use timestamped dirs for new runs |
| `--overwrite` | both | Re-run and replace existing results |
| `--base-url` | both | Desktop server |
| `--token` | both | Desktop server auth |

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
| Holo / Holotron model returns parser errors in `run_uitars.py` | Wrong runner for live model | Use `holo_run/run_holo.py` instead |
| `The model ui_tars_1.5 does not exist` | Run preflight; set `OPENAI_MODEL` to resolved id |
| vLLM 404 / connection refused | Restore SSH tunnel; verify `curl http://127.0.0.1:8010/health` |
| `APIConnectionError` mid-run | SSH tunnel dropped — restart tunnel, re-run failed tasks |
| Eval stopped when terminal closed | Eval was not in tmux — restart with `run_eval.sh start` |
| `osworld-eval` session already exists | Prior run still active — `run_eval.sh status` / `attach`; or `stop` then `start` |
| Need live logs | `run_eval.sh status` or `tmux attach -t osworld-eval` |
| `Token quota exceeded` (429) | Lower `--max-workers` or raise limit via `/set_token_limit` |
| Only 1 emulator despite `--max-workers 2` | Check `/status` — quota may be 1; call `/set_token_limit` |
| Stuck emulators after crash | `curl -X POST http://127.0.0.1:50003/stop_all_emulators` |
| Need free slots before parallel eval | Run `check_emulator_quota.sh`; stop running emulators if user confirms |

## Utility scripts

| Script | When |
|--------|------|
| `scripts/check_emulator_quota.sh` | **First** — before asking user for parallel count |
| `scripts/preflight.sh` | After user confirms count — health + model sanity + quota validation |
| `scripts/run_eval.sh` | **Start eval** — detached tmux `osworld-eval` (required unless user opts out) |

## Related skills

- **connect-dart-rollouter** — SSH tunnel in tmux `dart-rollouter-tunnel`
- **launch-dart-rollouter** — start model service on GPU VM
- **debug-osworld** — sequential rerun until first client error
- **analyze-rollouts** — per-task outcome tables from result dirs
