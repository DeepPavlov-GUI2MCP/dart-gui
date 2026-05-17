---
name: eval-coact
description: Run and customize the CoAct eval pipeline in GUI-Docker-Env. Use when working with `GUI-Docker-Env/run_coact.py`, CoAct eval configs, orchestrator/gui/coding backend settings, or CoAct eval mode behavior.
disable-model-invocation: true
---

# CoAct Eval

Use this skill when you need to run or customize the CoAct eval pipeline in `GUI-Docker-Env/run_coact.py`.

## Overview

The CoAct eval path is split into three independently configurable components:

- `orchestrator` — planning plus optional `web_search` / `read_webpage`
- `gui` — computer-only execution via the `computer` tool
- `coding` — optional coding subagent used when `enable_coding_agent` is on

The GUI component is computer-only in all modes. Research tools belong to the orchestrator side only.

## Startup Modes

`run_coact.py` supports exactly one of these modes per run:

- config mode: pass one fully specified config with `--config`
- args mode: pass the per-setting CLI flags directly

Do not mix `--config` with other `run_coact.py` flags.

## Config Layout

Keep committed example configs under:

- `GUI-Docker-Env/configs/coact/examples/openai.yml`
- `GUI-Docker-Env/configs/coact/examples/uitars_gui_agent.yml`

Copy examples into:

- `GUI-Docker-Env/configs/coact/local/`

Always copy a committed example into `GUI-Docker-Env/configs/coact/local/` before editing. Do not edit the committed example files in place.

Local copied configs are expected to be fully filled, standalone, and selected directly with `--config`.

When preparing a local config:

- Fill every backend value needed by the selected models directly in the copied config.
- Do not leave `api_key` or `base_url` blank if that backend requires them.
- Do not rely on `.env`, `.env-default`, or legacy args-mode fallbacks to silently provide missing config values.
- If a required value is missing, stop and ask the user for it.
- For OpenAI-hosted or OpenAI-compatible hosted models, ask for the API token if it is not already available.
- For self-hosted OpenAI-like models, ask for the base URL and token/API key if either is missing.

## Per-Component Customization

Customize each CoAct eval component independently:

- orchestrator: `model`, `base_url`, `api_key`
- gui: `model`, `base_url`, `api_key`
- coding: `model`, `base_url`, `api_key`

In config mode, put these under:

- `coact.orchestrator`
- `coact.gui`
- `coact.coding`

Config-mode backend entries should be explicit and complete for the run you are launching. If the user wants an OpenAI-backed run, write the actual OpenAI `base_url` and token into the local copied config. If the user wants a self-hosted backend, write the actual server URL and token into the local copied config.

In args mode, use:

- `--orchestrator_model`, `--orchestrator_base_url`, `--orchestrator_api_key`
- `--gui_model`, `--gui_base_url`, `--gui_api_key`
- `--coding_model`, `--coding_base_url`, `--coding_api_key`

Use config mode when you want multiple reusable setups with different private URLs or tokens.

## Mode Behavior

- `default`: GUI stays computer-only. Orchestrator research tools are only available if explicitly enabled.
- `search-first`: orchestrator gets `web_search` and `read_webpage`, and the prompt requires research before planning or GUI delegation.
- `inspect-source-first`: orchestrator gets `web_search` and `read_webpage`, but source inspection is optional guidance rather than a hard precondition.

## Desktop Emulator Tokens

- Use only `1` desktop emulator token for CoAct eval runs.
- Keep `--num_envs 1` for args-mode launches and keep config-mode concurrency at one emulator as well.
- If `0` desktop emulator tokens are free, stop and prompt the user to take action.
- Never kill active emulators without asking the user first.
- If capacity is full, ask whether the user wants to stop a specific emulator themselves, wants you to stop a specific emulator, or wants to wait.

## Usage

Config-mode example:

```bash
source .venv/bin/activate
cd GUI-Docker-Env
python run_coact.py \
  --config configs/coact/local/openai_thunderbird.yml
```

Args-mode example:

```bash
source .venv/bin/activate
cd GUI-Docker-Env
python run_coact.py \
  --task_id dfac9ee8-9bc4-4cdc-b465-4a4bfcd2f397 \
  --domain thunderbird \
  --test_config_base_dir evaluation_examples/examples \
  --mode search-first \
  --enable_coding_agent \
  --orchestrator_model gpt-5.5 \
  --orchestrator_base_url https://api.openai.com/v1 \
  --orchestrator_api_key "$OPENAI_API_KEY" \
  --gui_model gpt-5.5 \
  --gui_base_url https://api.openai.com/v1 \
  --gui_api_key "$OPENAI_API_KEY" \
  --coding_model gpt-5.1-mini \
  --coding_base_url https://api.openai.com/v1 \
  --coding_api_key "$OPENAI_API_KEY" \
  --num_envs 1
```

Use `--task_id` to run one OSWorld task directly by ID or by passing the path to a task JSON under `GUI-Docker-Env/evaluation_examples/examples/<domain>/`.

## Notes

- In config mode, metadata should reflect the resolved config values, not just the config path.
- The current `.env-default` values are legacy args-mode fallbacks, not the preferred way to vary CoAct eval backends.
- If you are documenting or discussing this pipeline, call it `CoAct eval`, not `search-enabled`.
- Treat emulator capacity as a user-controlled resource, not something the agent may reclaim automatically.
