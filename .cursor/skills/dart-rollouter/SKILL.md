---
name: dart-rollouter
description: Run the DART rollouter model service on a GPU VM in a dedicated tmux session, using minimal complementary Python requirements for an existing vLLM environment.
---

# DART Rollouter

Use this skill to launch the rollouter model service with UI-TARS-1.5 while reusing the machine's existing vLLM/Torch/CUDA stack.

## Tmux session (required)

**Default:** run `model_service.py` in a **dedicated** detached tmux session named `dart-rollouter` on the **GPU VM**. Do **not** run the long-lived service in the agent's current shell unless the user explicitly asks to (e.g. "run in foreground", "no tmux").

| Where | Session | Purpose |
|-------|---------|---------|
| GPU VM | `dart-rollouter` | `model_service.py` + vLLM — this skill |
| Local CPU | `dart-rollouter-tunnel` | SSH port-forward — **connect-dart-rollouter** |
| Local CPU | `osworld-eval` | OSWorld eval — **eval-osworld** |

Install and health checks run in the current shell; only `model_service.py` goes in `dart-rollouter`.

```bash
# On GPU VM — preferred (default UI-TARS-1.5 command)
bash /workspace/dart-gui/.cursor/skills/dart-rollouter/scripts/run_service.sh start

bash /workspace/dart-gui/.cursor/skills/dart-rollouter/scripts/run_service.sh status
bash /workspace/dart-gui/.cursor/skills/dart-rollouter/scripts/run_service.sh check
tmux attach -t dart-rollouter    # Ctrl+B then D to detach
bash /workspace/dart-gui/.cursor/skills/dart-rollouter/scripts/run_service.sh stop
```

**Opt-out:** If the user says to run in the foreground or without tmux, run `python model_service.py ...` directly in `validation/` and warn that closing the terminal will stop inference.

## Prerequisites

- GPU VM with NVIDIA runtime support.
- Repository checked out at `/workspace/dart-gui` (override with `REPO_ROOT` if different).
- `dart_rollouter/requirements_vllm_complementary.txt` present in the submodule.
- Existing system Python can import the target vLLM stack.
- `tmux` installed on the GPU VM.

## Install

Create a project virtual environment that can see the existing system packages, then install only the complementary rollouter dependencies (current shell — not tmux):

```bash
cd /workspace/dart-gui
uv venv --system-site-packages --python /usr/bin/python3.12 .venv
source .venv/bin/activate
uv pip install --no-deps -r dart_rollouter/requirements_vllm_complementary.txt
```

Use `--no-deps` so the project environment does not replace the existing vLLM, Torch, or CUDA packages.

## Launch

After install, start the service in tmux:

```bash
REPO_ROOT=/workspace/dart-gui bash "${REPO_ROOT}/.cursor/skills/dart-rollouter/scripts/run_service.sh" start
```

Custom Hydra overrides (optional):

```bash
SERVICE_CMD='source /workspace/dart-gui/.venv/bin/activate && cd /workspace/dart-gui/validation && python model_service.py ...' \
  bash /workspace/dart-gui/.cursor/skills/dart-rollouter/scripts/run_service.sh start
```

Default command (what `start` runs without `SERVICE_CMD`):

```bash
cd /workspace/dart-gui
source .venv/bin/activate
cd validation
python model_service.py \
  --config-name config_singleapp \
  model.ckpt_path=ByteDance-Seed/UI-TARS-1.5-7B \
  model.replicas=1 \
  model.base_port=8010 \
  model.host=0.0.0.0 \
  model.service_port=15961 \
  model.service_endpoint=http://localhost:15961 \
  model.vllm_params.gpu_memory_utilization=0.92 \
  +model.vllm_params.max_model_len=16384
```

## Verify

On the GPU VM after start (wait for vLLM to load):

```bash
bash /workspace/dart-gui/.cursor/skills/dart-rollouter/scripts/run_service.sh check
```

Or manually:

```bash
curl -s http://127.0.0.1:15961/status
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8010/health
```

Expected ports:

- `15961`: rollouter model service API.
- `8010`: vLLM OpenAI-compatible API.

From the local CPU machine, use **connect-dart-rollouter** to forward these ports before running **eval-osworld**.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Service stops when SSH disconnects | Was not in tmux — `run_service.sh start` |
| `dart-rollouter` session already exists | `run_service.sh status` / `attach`; or `stop` then `start` |
| Replace running session | `FORCE=1 run_service.sh start` |
| Health check fails right after start | vLLM still loading — `status` / `attach`, retry `check` |
| Missing `.venv` | Run **Install** first |

## Helper scripts

| Script | When |
|--------|------|
| `scripts/run_service.sh` | **Start/stop/status** model service in tmux `dart-rollouter` (required unless user opts out) |
| `no_docker.sh` | Thin wrapper → `run_service.sh start` |
| `docker.sh` | Optional legacy Docker-based bootstrap |

## Related skills

- **connect-dart-rollouter** — local SSH tunnel in tmux `dart-rollouter-tunnel`
- **eval-osworld** — local eval in tmux `osworld-eval` against forwarded vLLM
