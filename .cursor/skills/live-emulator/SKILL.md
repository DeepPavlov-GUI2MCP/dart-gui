---
name: live-emulator
description: Manage a live OSWorld emulator seeded from a task config for interactive debugging or agent-assisted experiments. Use when the user mentions a live emulator, seeded task image, task reset, manual GUI experimentation, or wants to interact with an emulator outside the eval runner.
---

# Live Emulator

Use this skill when you need a fresh emulator seeded from one OSWorld task config, but do not want to run the full eval pipeline.

## Scripts

All commands assume the repo root and an activated virtualenv:

```bash
source .venv/bin/activate
```

- `GUI-Docker-Env/scripts/live_emulator/manage_container.py`
- `GUI-Docker-Env/scripts/live_emulator/prepare_task.py`
- `GUI-Docker-Env/scripts/live_emulator/observe.py`
- `GUI-Docker-Env/scripts/live_emulator/act.py`
- `GUI-Docker-Env/scripts/live_emulator/window.py`
- `.cursor/skills/live-emulator/min-settings.json`

## Workflow

1. Start and seed a fresh emulator in one step:

```bash
python GUI-Docker-Env/scripts/live_emulator/manage_container.py start \
  --task chrome/0d8b7de3-e8de-4d86-b9fd-dd2dce58a217
```

2. If you already have a running emulator, seed or reseed it explicitly:

```bash
python GUI-Docker-Env/scripts/live_emulator/prepare_task.py \
  --latest \
  --task chrome/0d8b7de3-e8de-4d86-b9fd-dd2dce58a217
```

3. Observe or interact:

```bash
python GUI-Docker-Env/scripts/live_emulator/observe.py \
  --emulator-id <id> \
  --output /tmp/live.png
```

```bash
python GUI-Docker-Env/scripts/live_emulator/act.py \
  --emulator-id <id> \
  --pyautogui "pyautogui.click(500, 400)"
```

4. Inspect or randomize window geometry using the skill-local threshold file:

```bash
python GUI-Docker-Env/scripts/live_emulator/window.py bounds \
  --emulator-id <id> \
  --task thunderbird/08c73485-7c6d-4681-999d-919f5c32dcfa
```

```bash
python GUI-Docker-Env/scripts/live_emulator/window.py randomize \
  --emulator-id <id> \
  --task thunderbird/08c73485-7c6d-4681-999d-919f5c32dcfa
```

```bash
python GUI-Docker-Env/scripts/live_emulator/window.py resize \
  --emulator-id <id> \
  --task thunderbird/08c73485-7c6d-4681-999d-919f5c32dcfa \
  --width 1320 \
  --height 900
```

5. Reset to a fresh container when you want the same task from clean state again:

```bash
python GUI-Docker-Env/scripts/live_emulator/manage_container.py reset \
  --emulator-id <id> \
  --task chrome/0d8b7de3-e8de-4d86-b9fd-dd2dce58a217
```

## Notes

- Prefer `manage_container.py start --task ...` for the common fresh-start workflow.
- Keep `prepare_task.py` for reseeding an already-running emulator or applying a different task later.
- `window.py` uses `.cursor/skills/live-emulator/min-settings.json` to resolve app or task-specific minimum sizes.
- Max X and max Y are computed dynamically from the current screen size and chosen window size, not stored as fixed values.
- `bounds` shows the resolved threshold and safe coordinate range before you apply changes.
- `move` and `resize` clamp by default so the window stays onscreen and above the configured minimum size.
- `reset` means stop the current Docker-backed emulator, start a fresh one, and optionally reseed the task. The current Docker provider does not support in-place image reload.
- Prefer one task per fresh emulator when comparing repeated attempts.
- Task proxy setup is off by default. Only pass `--enable-task-proxy` when you explicitly want the task's proxy behavior.
- The scripts return JSON by default. Use `--format text` for a terse human-readable output.
