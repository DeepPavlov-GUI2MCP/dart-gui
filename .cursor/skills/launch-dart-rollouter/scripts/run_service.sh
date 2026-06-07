#!/usr/bin/env bash
# Launch model_service.py in a dedicated tmux session (local GPU machine).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

FORCE="${FORCE:-}"

usage() {
  cat <<EOF
Usage: $(basename "$0") <start|stop|restart|status|attach|check|check-venv|list|install> [extra hydra overrides...]

  start       Run model_service.py in tmux (DEVICE + PROFILE from manifest)
  stop        Kill tmux session for the selected device
  restart     stop then start
  status      Session state and recent log tail
  attach      Attach to tmux session
  check       curl model service + vLLM health on this machine
  check-venv  Verify .venv has vllm/torch and CUDA GPUs
  list        Show devices and profiles from manifest.local.yaml
  install     Create or verify .venv for DEVICE/PROFILE

Environment:
  DEVICE=${DEVICE:-<manifest default>}
  PROFILE=${PROFILE:-<manifest default>}
  SERVICE_CMD   Override full launch command (skips profile resolution)
  FORCE=1       Replace existing tmux session on start
  DART_ROLLOUTER_MANIFEST  Override manifest path (default: config/manifest.local.yaml)

Examples:
  PROFILE=holo3-2gpu $(basename "$0") start
  DEVICE=rollouter-gpu2 PROFILE=uitars-1.5 $(basename "$0") start
  $(basename "$0") list
EOF
}

resolve_service_cmd() {
  if [[ -n "${SERVICE_CMD:-}" ]]; then
    printf '%s' "${SERVICE_CMD}"
    return
  fi
  load_manifest
  local cmd="${SERVICE_CMD}"
  if [[ $# -gt 0 ]]; then
    cmd="${cmd} $*"
  fi
  printf '%s' "${cmd}"
}

require_prereqs() {
  load_manifest
  if ! command -v tmux >/dev/null 2>&1; then
    echo "FAIL: tmux is required but not installed" >&2
    exit 1
  fi
  if [[ ! -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    echo "FAIL: missing virtualenv: ${REPO_ROOT}/.venv/bin/activate" >&2
    echo "Run: DEVICE=${DEVICE} PROFILE=${PROFILE} $(basename "$0") install" >&2
    exit 1
  fi
}

start_service() {
  require_prereqs
  local service_cmd
  service_cmd="$(resolve_service_cmd "$@")"

  echo "Device:  ${DEVICE}  Profile: ${PROFILE}"
  echo "Session: ${SESSION_NAME}  Repo: ${REPO_ROOT}"
  if [[ "${TENSOR_PARALLEL_SIZE:-1}" != "1" ]]; then
    echo "Tensor parallel: ${TENSOR_PARALLEL_SIZE} GPUs per replica"
  fi

  if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    if [[ "${FORCE}" == "1" ]]; then
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
}

stop_service() {
  load_manifest
  if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    tmux kill-session -t "${SESSION_NAME}"
    echo "Stopped tmux session: ${SESSION_NAME}"
  else
    echo "No tmux session: ${SESSION_NAME}"
  fi
}

show_status() {
  load_manifest
  echo "Device: ${DEVICE}  Profile: ${PROFILE}"
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
  load_manifest
  if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "FAIL: no session ${SESSION_NAME}. Start with: $(basename "$0") start" >&2
    exit 1
  fi
  exec tmux attach -t "${SESSION_NAME}"
}

check_health() {
  load_manifest
  echo "== Model service: http://127.0.0.1:${MODEL_PORT}/status =="
  if curl -sf --connect-timeout 5 "http://127.0.0.1:${MODEL_PORT}/status"; then
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
  check-venv) check_venv ;;
  list) list_manifest ;;
  install) install_venv ;;
  -h|--help|help|"") usage ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    usage >&2
    exit 1
    ;;
esac
