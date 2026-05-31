---
name: eval-coact
description: Run and customize the CoAct eval pipeline in GUI-Docker-Env. Use for `run_coact.py`, CoAct configs, orchestrator/gui/coding backends, Codex+MCP CLI rollouts, and capturing synthetic dataset trajectories from prepared task JSONs.
disable-model-invocation: true
---

# CoAct Eval

Use this skill for `GUI-Docker-Env/run_coact.py`: standard OSWorld evals, orchestrated CoAct runs, and **synthetic dataset rollouts** (Codex agent + MCP `computer` tool over prepared task/plan JSONs).

## Overview

Three independently configurable components:

- `orchestrator` — planning plus optional `web_search` / `read_webpage`
- `gui` — computer-only execution (`openai` CUA, `vllm` UI-TARS, or `cli` Codex+MCP)
- `coding` — optional coding subagent when `enable_coding_agent` is on

Research tools belong to the orchestrator only. The GUI path is computer-only in all modes.

## Synthetic dataset rollouts (primary workflow)

Use this when a task JSON already contains the full GUI plan in `instruction` (e.g. Thunderbird `exponential_50_20` train tasks). Skip the orchestrator and run Codex with MCP against the live emulator.

### Dataset layout

Tasks and rollouts live under a named dataset directory (e.g. `exponential_50_20`):

```
{datasets_root}/{dataset_name}/
  split_golden.csv | split_train.csv | split_test.csv
  golden/tasks/golden.json
  train/tasks/train_0001.json …
  test/tasks/test_0001.json …
  {golden|train|test}/rollouts/{task_id}/{rollout_id}/
```

- **Tasks:** OSWorld-shaped JSON (`id`, `snapshot`, `instruction`, `config`, `evaluator`). The `instruction` field is the agent prompt (often a step-by-step plan).
- **Rollouts:** one timestamped folder per CoAct run. `rollout_id` defaults to UTC `YYYYMMDDTHHMMSSZ`; override with `--rollout-id`.

`run_coact.py` detects synthetic layout when `--task_id` points at `…/{split}/tasks/{task_id}.json` (`split` ∈ `golden`, `train`, `test`). Artifacts are written under `…/{split}/rollouts/{task_id}/{rollout_id}/`, not under `--result_dir/coact/…`.

Example dataset root:

`synthetic_data/thunderbird/08c73485-7c6d-4681-999d-919f5c32dcfa/datasets/exponential_50_20/`

### Prepare tasks

Regenerate task JSONs after CSV edits:

```bash
python synthetic_data/thunderbird/08c73485-7c6d-4681-999d-919f5c32dcfa/scripts/generate_pref_tasks.py
```

One-time migration from legacy flat layout (`train/train_0001.json`) to `train/tasks/`:

```bash
python synthetic_data/thunderbird/08c73485-7c6d-4681-999d-919f5c32dcfa/scripts/group_dataset_tasks.py
```

### Prerequisites

1. **Desktop server** on `http://localhost:50003` (`curl -s http://localhost:50003/ping`).
2. **One free emulator token** — use `--num_envs 1`. If quota is full, ask the user before stopping emulators.
3. **Codex CLI** on PATH with working `~/.codex` auth (`codex` required when `gui.protocol: cli`).
4. **Venv:** `source GUI-Docker-Env/.venv/bin/activate` (or `/mnt/dart-gui/.venv` per repo layout).

### Canonical single-task command

```bash
source GUI-Docker-Env/.venv/bin/activate
cd GUI-Docker-Env

python run_coact.py \
  --no-orchestrator \
  --gui_protocol cli \
  --gui_model gpt-5.4 \
  --task_id ../synthetic_data/thunderbird/08c73485-7c6d-4681-999d-919f5c32dcfa/datasets/exponential_50_20/train/tasks/train_0001.json \
  --domain thunderbird \
  --provider_name docker_server \
  --num_envs 1 \
  --cua_max_steps 50 \
  --gui-cli-timeout-seconds 1200
```

Notes:

- `--domain` can be `thunderbird` or `all`; domain/id are also read from the JSON (`snapshot`, `id`).
- `--result_dir` only affects legacy paths and top-level `results_coact/coact/run_metadata_*.json`; synthetic rollouts still land under `…/rollouts/…`.
- Increase `--cua_max_steps` for multi-pref tasks (e.g. 50 for `train_0026`).
- Re-run the same `rollout_id` only if you intend to overwrite that folder; each new run gets a new timestamp by default.

### Batch rollouts

Loop over task files (one emulator at a time unless the user approves parallelism):

```bash
DATASET=../synthetic_data/thunderbird/08c73485-7c6d-4681-999d-919f5c32dcfa/datasets/exponential_50_20
for task in "$DATASET"/train/tasks/train_00{01..10}.json; do
  python run_coact.py --no-orchestrator --gui_protocol cli --gui_model gpt-5.4 \
    --task_id "$task" --domain thunderbird --provider_name docker_server \
    --num_envs 1 --cua_max_steps 50 --gui-cli-timeout-seconds 1200
done
```

Prefer subagents or explicit task lists for large splits; verify `result.txt` and evaluator score after each run.

### Rollout artifacts (CLI + MCP)

Under `{split}/rollouts/{task_id}/{rollout_id}/`:

| File | Role |
|------|------|
| `responses.json` | Per-turn model transcript for SFT conversion (`turn_index`, `agent_message`, `actions`, `call_id`) plus `type: "final"` |
| `tool_events.json` | Action-only `computer_call` log |
| `cli_turn_0001.jsonl` | Codex event stream (ground-truth order for pairing) |
| `cli_turn_0001.txt` | Human-readable Codex log |
| `step_0001.png` … | Screenshots (initial + one per tool step) |
| `history_inputs.json` | Harness message list |
| `result.txt` | Evaluator score (`1.0` = pass) |
| `metadata.json` | Run settings; `dataset.rollout_dir`, `rollout_id`, `split` |
| `planning/instruction.txt` | Exact prompt sent to Codex |
| `.codex/` | Isolated `CODEX_HOME` for this rollout |

### `responses.json` pairing (CLI)

Built chronologically from `cli_turn_0001.jsonl` + `tool_events.json`:

1. Walk jsonl `item.completed` events in order.
2. Each non-terminal `agent_message` pairs with the **next** `computer` tool call → one tool turn.
3. Terminal-only messages (`TERMINATE`, `IDK`, or short terminate-only text) → **`type: "final"` only**, not merged into a tool turn.
4. Back-to-back `computer` calls without a new message → empty `agent_message` on the extra turn.

Codex is instructed to finish GUI work first, then send a message-only `TERMINATE` with no further `computer` calls. Use `cli_turn_0001.jsonl` to debug pairing; do not train on tool turns after a terminal message.

### Config-mode equivalent

Copy `configs/coact/examples/no_orchestrator.yml` → `configs/coact/local/`, set `gui.protocol: cli`, `gui.model`, API keys, `runtime.provider_name: docker_server`, and `tasks.task_id` to the full path under `…/tasks/`. Launch with `--config` only (no other CLI flags).

## CLI GUI (Codex + MCP `computer` tool)

**Pattern B:** Codex calls a stdio MCP server (`mm_agents/coact/desktop_computer_mcp.py`) exposing a `computer` tool. The server executes OpenAI CUA-style action batches on the VM and returns screenshots.

- Per-rollout `CODEX_HOME` under `<save_path>/.codex/` with isolated `config.toml`.
- Codex runs with `--dangerously-bypass-approvals-and-sandbox` for non-interactive `exec`.
- `.rules` in the rollout directory discourages shell `command_execution`; GUI should use MCP only.
- `--gui-cli-timeout-seconds` (default 300) — use 900–1200 for full Thunderbird pref tasks.

Requires `codex` on PATH. Validate with a single synthetic task before batching.

## No orchestrator (`--no-orchestrator`)

Skips orchestrator for all `gui.protocol` values. Task `instruction` goes directly to the GUI agent (one env loop, or multi-rollout grid if `script_mode: multi-rollout`).

For synthetic datasets, always pair with `gui.protocol: cli` unless the user explicitly wants OpenAI CUA or vLLM UI-TARS.

## Startup modes

Exactly one per run:

- **Config mode:** `--config path/to/local.yml` (no other `run_coact.py` flags).
- **Args mode:** CLI flags only.

## Config layout

Examples (do not edit in place):

- `GUI-Docker-Env/configs/coact/examples/openai.yml`
- `GUI-Docker-Env/configs/coact/examples/uitars_gui_agent.yml`
- `GUI-Docker-Env/configs/coact/examples/no_orchestrator.yml`

Copy to `configs/coact/local/`, fill all required `api_key` / `base_url` values, then run with `--config`.

## Per-component customization

| Component | Config keys | Args mode |
|-----------|-------------|-----------|
| orchestrator | `coact.orchestrator.*` | `--orchestrator_model`, `_base_url`, `_api_key` |
| gui | `coact.gui.*` (`protocol`, `model`, …) | `--gui_model`, `--gui_protocol`, `--gui_base_url`, `--gui_api_key`, `--openai-force-completions-api` |
| coding | `coact.coding.*` | `--coding_model`, … |

## OpenAI GUI: Completions vs Responses

When `gui.protocol` is `openai` (default), CoAct uses **`/chat/completions` with a `computer` function tool** by default (`--openai-force-completions-api`, enabled unless you pass `--no-openai-force-completions-api`). This works with OpenRouter-style and other chat-only proxies.

- **Default:** in-process DesktopEnv + MCP-compatible CUA action JSON via function calling.
- **Opt out:** `--no-openai-force-completions-api` uses native OpenAI **`/responses`** + built-in `{type: "computer"}` (requires provider support).
- **`cli`:** still uses Codex subprocess + stdio MCP (unchanged).
- **`vllm`:** UI-TARS text actions, no computer tool (unchanged).

YAML: `coact.gui.openai_force_completions_api: true|false`

## Mode behavior

- `default`: GUI computer-only; orchestrator research only if enabled.
- `search-first`: orchestrator must research before planning/GUI.
- `inspect-source-first`: research tools on; source inspection encouraged.

## Desktop emulator tokens

- Use **one** emulator: `--num_envs 1`.
- If **0** tokens free, stop and ask the user — do not kill emulators without permission.

## Standard OSWorld eval (non-synthetic)

```bash
source GUI-Docker-Env/.venv/bin/activate
cd GUI-Docker-Env
python run_coact.py \
  --task_id dfac9ee8-9bc4-4cdc-b465-4a4bfcd2f397 \
  --domain thunderbird \
  --test_config_base_dir evaluation_examples/examples \
  --mode search-first \
  --orchestrator_model gpt-5.5 --gui_model gpt-5.5 \
  --num_envs 1
```

Results go to `--result_dir/coact/{domain}/{task_id}/` unless the task path uses the synthetic `…/tasks/` layout.

## Token spending metadata

CoAct records LLM token usage and estimated cost in existing metadata files (not in trace files). There is no `cost.txt`.

| Level | File | `spending` section |
|-------|------|-------------------|
| Attempt | `{attempt_dir}/metadata.json` | Per multi-rollout GUI attempt |
| Task | `{history_save_dir}/metadata.json` | Aggregated attempts (+ planning when orchestrator multi-rollout) |
| Run | `{result_dir}/coact/run_metadata_latest.json` | Recomputed from all tasks in eval scope on every invocation |

Each `spending` block includes `spending_per_step`, `spending_by_model`, `spending_by_role`, and `spending_total`. Multi-attempt tasks also have `spending_by_attempt`; multi-task runs also have `spending_by_task`, `tasks_included`, and `tasks_missing_spending`.

Run-level totals are always refreshed after the worker pool (or on the “no tasks to process” path) by reading current task-root `metadata.json` files for all selected tasks — partial reruns overwrite only the rerun task’s contribution.

Credentials: `OPENROUTER_*` when set, else `OPENAI_*` from `.env`. Local vLLM (`127.0.0.1:8010`) tracks tokens with `$0` cost.

## Notes

- Call this pipeline **CoAct eval**, not `search-enabled`.
- Config-mode metadata should record resolved backend values, not only the config path.
- `.env-default` is a legacy args-mode fallback; prefer explicit local configs for reproducible runs.
- Related: dataset README under `synthetic_data/…/datasets/<name>/README.md`; task generator `scripts/generate_pref_tasks.py`; path helper `mm_agents/coact/synthetic_rollout_paths.py`.
