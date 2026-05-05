---
name: dart-rollouter
description: Run the DART rollouter model service on a GPU VM using the minimal complementary Python requirements for an existing vLLM environment.
---

# DART Rollouter

Use this skill to launch the rollouter model service with UI-TARS-1.5 while reusing the machine's existing vLLM/Torch/CUDA stack.

## Prerequisites

- GPU VM with NVIDIA runtime support.
- Repository checked out at `/workspace/dart-gui`.
- `dart_rollouter/requirements_vllm_complementary.txt` present in the submodule.
- Existing system Python can import the target vLLM stack.

## Install

Create a project virtual environment that can see the existing system packages, then install only the complementary rollouter dependencies:

```bash
cd /workspace/dart-gui
uv venv --system-site-packages --python /usr/bin/python3.12 .venv
source .venv/bin/activate
uv pip install --no-deps -r dart_rollouter/requirements_vllm_complementary.txt
```

Use `--no-deps` so the project environment does not replace the existing vLLM, Torch, or CUDA packages.

## Launch

Run the model service from `validation/`:

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

For a persistent background session:

```bash
tmux new-session -d -s dart-rollouter "bash -lc 'source /workspace/dart-gui/.venv/bin/activate && cd /workspace/dart-gui/validation && python model_service.py --config-name config_singleapp model.ckpt_path=ByteDance-Seed/UI-TARS-1.5-7B model.replicas=1 model.base_port=8010 model.host=0.0.0.0 model.service_port=15961 model.service_endpoint=http://localhost:15961 model.vllm_params.gpu_memory_utilization=0.92 +model.vllm_params.max_model_len=16384'"
```

## Verify

Health checks on the VM:

```bash
curl -s http://127.0.0.1:15961/status
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8010/health
```

Expected ports:

- `15961`: rollouter model service API.
- `8010`: vLLM OpenAI-compatible API.
