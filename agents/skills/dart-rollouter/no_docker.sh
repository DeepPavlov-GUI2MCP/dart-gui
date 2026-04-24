#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/workspace/dart-gui"
SESSION_NAME="dart-rollouter"
VENV_ACTIVATE="${REPO_ROOT}/.venv/bin/activate"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but not installed"
  exit 1
fi

if [ ! -f "${VENV_ACTIVATE}" ]; then
  echo "missing virtualenv activate script: ${VENV_ACTIVATE}"
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  exit 0
fi

RUN_CMD="source ${VENV_ACTIVATE} && cd ${REPO_ROOT}/validation && python model_service.py --config-name config_singleapp model.ckpt_path=ByteDance-Seed/UI-TARS-1.5-7B model.replicas=1 model.base_port=8010 model.host=0.0.0.0 model.service_port=15961 model.vllm_params.gpu_memory_utilization=0.92 +model.vllm_params.max_model_len=4096"
tmux new-session -d -s "${SESSION_NAME}" "bash -lc '${RUN_CMD}'"
