#!/usr/bin/env bash
# Launch model_service.py in a dedicated tmux session on the GPU VM.
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-dart-rollouter}"
REPO_ROOT="${REPO_ROOT:-/workspace/dart-gui}"
MODEL_SERVICE_PORT="${MODEL_SERVICE_PORT:-15961}"
VLLM_PORT="${VLLM_PORT:-8010}"

default_service_cmd() {
  printf '%s' "source ${REPO_ROOT}/.venv/bin/activate && cd ${REPO_ROOT}/validation && python model_service.py \
  --config-name config_singleapp \
  model.ckpt_path=ByteDance-Seed/UI-TARS-1.5-7B \
  model.replicas=1 \
  model.base_port=${VLLM_PORT} \
  model.host=0.0.0.0 \
  model.service_port=${MODEL_SERVICE_PORT} \
  model.service_endpoint=http://localhost:${MODEL_SERVICE_PORT} \
  model.vllm_params.gpu_memory_utilization=0.92 \
  +model.vllm_params.max_model_len=16384"
}

usage() {
  cat <<EOF
Usage: $(basename "$0") <start|stop|restart|status|attach|check> [service command...]

  start    Run model_service.py in detached tmux session "${SESSION_NAME}"
           Uses built-in UI-TARS-1.5 command unless SERVICE_CMD is set or args follow "start".
  stop     Kill tmux session "${SESSION_NAME}"
  restart  stop then start
  status   Show session state and recent log tail
  attach   Attach to session (Ctrl+B then D to detach)
  check    curl health for model service and vLLM on this VM

Environment:
  REPO_ROOT=${REPO_ROOT}
  SESSION_NAME, SERVICE_CMD, FORCE=1 (replace existing session on start)
EOF
}

resolve_service_cmd() {
  local cmd="${SERVICE_CMD:-}"
  if [[ $# -gt 0 ]]; then
    cmd="$*"
  fi
  if [[ -z "${cmd}" ]]; then
    cmd="$(default_service_cmd)"
  fi
  printf '%s' "${cmd}"
}

require_prereqs() {
  if ! command -v tmux >/dev/null 2>&1; then
    echo "FAIL: tmux is required but not installed" >&2
    exit 1
  fi
  if [[ ! -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    echo "FAIL: missing virtualenv: ${REPO_ROOT}/.venv/bin/activate" >&2
    exit 1
  fi
}

start_service() {
  require_prereqs
  local service_cmd
  service_cmd="$(resolve_service_cmd "$@")"

  if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    if [[ "${FORCE:-}" == "1" ]]; then
      tmux kill-session -t "${SESSION_NAME}"
    else
      echo "FAIL: tmux session ${SESSION_NAME} already exists (service may still be running)" >&2
      echo "  status: $(basename "$0") status" >&2
      echo "  attach: $(basename "$0") attach" >&2
      echo "  stop:   $(basename "$0") stop" >&2
      echo "  retry:  FORCE=1 $(basename "$0") start" >&2
      exit 1
    fi
  fi

  tmux new-session -d -s "${SESSION_NAME}" -n model "bash -lc $(printf '%q' "${service_cmd}")"
  sleep 1

  if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "FAIL: tmux session ${SESSION_NAME} did not start" >&2
    exit 1
  fi

  echo "Started model service in tmux session: ${SESSION_NAME}"
  echo "Attach:  tmux attach -t ${SESSION_NAME}"
  echo "Status:  $(basename "$0") status"
  echo "Check:   $(basename "$0") check"
  echo "Repo:    ${REPO_ROOT}"
}

stop_service() {
  if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    tmux kill-session -t "${SESSION_NAME}"
    echo "Stopped tmux session: ${SESSION_NAME}"
  else
    echo "No tmux session: ${SESSION_NAME}"
  fi
}

show_status() {
  if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "tmux session: ${SESSION_NAME} (running)"
    tmux list-windows -t "${SESSION_NAME}" -F '  window: #{window_name} (#{pane_current_command})' 2>/dev/null || true
    echo
    echo "Recent pane output (last 30 lines):"
    tmux capture-pane -t "${SESSION_NAME}" -p -S -30 2>/dev/null || true
  else
    echo "tmux session: ${SESSION_NAME} (not running)"
  fi
}

attach_service() {
  if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "FAIL: no session ${SESSION_NAME}. Start with: $(basename "$0") start" >&2
    exit 1
  fi
  exec tmux attach -t "${SESSION_NAME}"
}

check_health() {
  echo "== Model service: http://127.0.0.1:${MODEL_SERVICE_PORT}/status =="
  if curl -sf --connect-timeout 5 "http://127.0.0.1:${MODEL_SERVICE_PORT}/status"; then
    echo
    echo "OK"
  else
    echo "FAIL: model service not reachable" >&2
    exit 1
  fi
  echo
  echo "== vLLM: http://127.0.0.1:${VLLM_PORT}/health =="
  local code
  code="$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 5 "http://127.0.0.1:${VLLM_PORT}/health" || true)"
  if [[ "${code}" == "200" ]]; then
    echo "OK (HTTP ${code})"
  else
    echo "FAIL: vLLM health returned HTTP ${code:-none}" >&2
    exit 1
  fi
}

cmd="${1:-}"
shift || true

case "${cmd}" in
  start) start_service "$@" ;;
  stop) stop_service ;;
  restart)
    stop_service
    start_service "$@"
    ;;
  status) show_status ;;
  attach) attach_service ;;
  check) check_health ;;
  -h|--help|help|"") usage ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    usage >&2
    exit 1
    ;;
esac
