---
name: launch-dart-rollouter
description: Run the DART rollouter model service on one or more GPU devices in dedicated tmux sessions, using device/profile manifests and minimal complementary Python requirements for an existing vLLM environment. Supports UI-TARS-1.5 and Holo3-35B (2-GPU tensor parallel).
---

# Launch dart-rollouter

Launch `model_service.py` + vLLM on GPU machine(s). Devices and model presets are resolved from **`config/manifest.local.yaml`** (gitignored, machine-specific). Committed defaults live in **`config/manifest.defaults.yaml`**. On first launch, `manifest.local.yaml` is copied from defaults if missing. **Repo root auto-detects** from the skill path — no hardcoded `/workspace/dart-gui` required on the primary host.

## Quick start (this GPU host)

From the repo root:

```bash
# Verify bundled .venv (vLLM 0.19.1 + 2x A100)
bash .cursor/skills/launch-dart-rollouter/scripts/run_service.sh check-venv

# Holo3 on two GPUs
PROFILE=holo3-2gpu bash .cursor/skills/launch-dart-rollouter/scripts/run_service.sh start
PROFILE=holo3-2gpu bash .cursor/skills/launch-dart-rollouter/scripts/run_service.sh check
tmux attach -t dart-rollouter    # Ctrl+B then D to detach
```

**UI-TARS-1.5** (default profile):

```bash
bash .cursor/skills/launch-dart-rollouter/scripts/run_service.sh start
```

List devices and profiles:

```bash
bash .cursor/skills/launch-dart-rollouter/scripts/run_service.sh list
```

## Remote launch (from CPU / laptop)

When `ssh_host` is set on a device entry:

```bash
PROFILE=holo3-2gpu bash .cursor/skills/launch-dart-rollouter/scripts/remote.sh start
PROFILE=holo3-2gpu bash .cursor/skills/launch-dart-rollouter/scripts/remote.sh check
```

## Tmux session (required)

**Default:** run `model_service.py` in a **dedicated** detached tmux session on the **GPU VM**. Do **not** run the long-lived service in the agent's current shell unless the user explicitly asks to (e.g. "run in foreground", "no tmux").

| Where | Session | Purpose |
|-------|---------|---------|
| GPU VM | `dart-rollouter` (or per-device name in manifest) | `model_service.py` + vLLM — this skill |
| Local CPU | `dart-rollouter-tunnel` | SSH port-forward — **connect-dart-rollouter** |
| Local CPU | `osworld-eval` | OSWorld eval — **eval-osworld** |

```bash
bash .cursor/skills/launch-dart-rollouter/scripts/run_service.sh start
bash .cursor/skills/launch-dart-rollouter/scripts/run_service.sh status
bash .cursor/skills/launch-dart-rollouter/scripts/run_service.sh check
bash .cursor/skills/launch-dart-rollouter/scripts/run_service.sh stop
```

**Opt-out:** If the user says to run in the foreground or without tmux, run the resolved `SERVICE_CMD` directly in `validation/` and warn that closing the terminal will stop inference.

## Manifest layout

| File | Tracked | Purpose |
|------|---------|---------|
| `config/manifest.defaults.yaml` | yes | Committed template — safe defaults for new machines |
| `config/manifest.local.yaml` | no (gitignored) | **Active launch config** — edit replicas, `max_num_seqs`, ports, etc. |

**First launch** copies defaults → local automatically.

**Tune this machine:**

```bash
# Edit live settings (e.g. holotron replicas, max_num_seqs)
${EDITOR:-nano} .cursor/skills/launch-dart-rollouter/config/manifest.local.yaml

# Reset local manifest from committed defaults
cp .cursor/skills/launch-dart-rollouter/config/manifest.defaults.yaml \
   .cursor/skills/launch-dart-rollouter/config/manifest.local.yaml
```

Override path entirely: `DART_ROLLOUTER_MANIFEST=/path/to/manifest.yaml`.

## Devices and profiles

Edit **`manifest.local.yaml`** for machine-specific launch settings. Update **`manifest.defaults.yaml`** when adding new profiles/devices for the whole team.

| Variable | Purpose |
|----------|---------|
| `DEVICE` | Host entry in manifest (default: `dart-rollouter`) |
| `PROFILE` | Model preset (default: `uitars-1.5`) |
| `REPO_ROOT` | Override auto-detected repo path |
| `FORCE=1` | Replace existing tmux session on start |
| `SERVICE_CMD` | Override full launch command |
| `DART_ROLLOUTER_MANIFEST` | Override manifest path (default: `manifest.local.yaml`) |

**Built-in profiles**

| Profile | Model | GPUs | Notes |
|---------|-------|------|-------|
| `uitars-1.5` | `ByteDance-Seed/UI-TARS-1.5-7B` | 1 | Default |
| `holo3-2gpu` | `Hcompany/Holo3-35B-A3B` | 2 (TP=2) | One replica spans both GPUs |
| `holotron-3-nano` | `Hcompany/Holotron-3-Nano` | 2 (TP=1) | One replica per GPU; tune `max_num_seqs` in local manifest |

**Built-in devices**

| Device | Layout | venv mode | Ports (vLLM / pool) |
|--------|--------|-----------|---------------------|
| `dart-rollouter` | Primary host (auto repo root) | `bundled` — full vLLM in `.venv` | 8010 / 15961 |
| `workspace` | `/workspace/dart-gui` Docker VM | `complementary` — reuses system vLLM | 8010 / 15961 |
| `rollouter-gpu2` | Second GPU host via SSH | `complementary` | 8000 / 15959 |

## Prerequisites

- GPU machine with NVIDIA drivers and `tmux`.
- **Primary host (`dart-rollouter`)**: bundled `.venv` at repo root with vLLM ≥ 0.19 (Holo3).
- **Legacy / Docker hosts (`workspace`)**: system Python with vLLM + complementary `--no-deps` install.

## Install / verify venv

**Primary host** — venv already exists; just verify:

```bash
bash .cursor/skills/launch-dart-rollouter/scripts/run_service.sh check-venv
```

If missing, create bundled venv (installs vLLM + deps from `requirements_cuda_128_complementary.txt`):

```bash
PROFILE=holo3-2gpu bash .cursor/skills/launch-dart-rollouter/scripts/run_service.sh install
```

**Legacy complementary layout** (system-site-packages):

```bash
DEVICE=workspace bash .cursor/skills/launch-dart-rollouter/scripts/run_service.sh install
```

Manual equivalent for bundled venv:

```bash
cd "$(git rev-parse --show-toplevel)"
uv venv --python python3.11 .venv
source .venv/bin/activate
uv pip install -r dart_rollouter/requirements_cuda_128_complementary.txt
python -c "import vllm; print(vllm.__version__)"
```

## Launch

Resolved Holo3 command (what `PROFILE=holo3-2gpu start` runs):

```bash
source .venv/bin/activate
cd validation
python model_service.py \
  --config-name config_singleapp \
  model.ckpt_path=Hcompany/Holo3-35B-A3B \
  model.replicas=1 \
  model.base_port=8010 \
  model.host=0.0.0.0 \
  model.service_port=15961 \
  model.service_endpoint=http://localhost:15961 \
  +model.vllm_params.dtype=bfloat16 \
  +model.vllm_params.gpu_memory_utilization=0.90 \
  +model.vllm_params.max_model_len=8192 \
  +model.vllm_params.max_num_seqs=1 \
  +model.vllm_params.tensor_parallel_size=2
```

Inspect resolved command without starting:

```bash
python3 .cursor/skills/launch-dart-rollouter/scripts/resolve_config.py -p holo3-2gpu --format cmd
```

## Verify

After start (wait for vLLM to load — Holo3 may take several minutes):

```bash
PROFILE=holo3-2gpu bash .cursor/skills/launch-dart-rollouter/scripts/run_service.sh check
```

From the local CPU machine, use **connect-dart-rollouter** before running **eval-osworld**.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Service stops when SSH disconnects | Was not in tmux — `run_service.sh start` |
| Session already exists | `status` / `attach`; or `stop` then `start` |
| Replace running session | `FORCE=1 ... start` |
| Health check fails right after start | vLLM still loading — `status` / `attach`, retry `check` |
| Missing vllm in .venv | `check-venv`; then `install` |
| Holo3 OOM on one GPU | Use `PROFILE=holo3-2gpu` (TP=2), not `replicas=2` |
| Wrong repo path | `run_service.sh list` shows auto-detected root; set `REPO_ROOT` to override |

## Helper scripts

| Script | When |
|--------|------|
| `config/manifest.defaults.yaml` | Committed device/profile template |
| `config/manifest.local.yaml` | Gitignored active launch manifest |
| `scripts/resolve_config.py` | Resolve DEVICE/PROFILE → launch settings |
| `scripts/run_service.sh` | Start/stop/status on **local** GPU machine |
| `scripts/remote.sh` | Same commands over SSH to manifest device |
| `no_docker.sh` | Back-compat wrapper → `run_service.sh start` |

## Related skills

- **connect-dart-rollouter** — local SSH tunnel in tmux `dart-rollouter-tunnel`
- **eval-osworld** — local eval in tmux `osworld-eval` against forwarded vLLM
