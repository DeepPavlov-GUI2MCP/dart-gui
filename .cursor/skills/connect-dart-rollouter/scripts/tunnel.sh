#!/usr/bin/env bash
# Manage SSH port-forward tunnel to dart-rollouter GPU VM.
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-dart-rollouter-tunnel}"
SSH_HOST="${SSH_HOST:-dart-rollouter}"

# Remote ports (discover on VM if these change).
REMOTE_VLLM_PORT="${REMOTE_VLLM_PORT:-8000}"
REMOTE_MODEL_PORT="${REMOTE_MODEL_PORT:-15959}"

# Local aliases used by GUI-Docker-Env (.env OPENAI_BASE_URL=http://127.0.0.1:8010).
LOCAL_VLLM_ALIAS="${LOCAL_VLLM_ALIAS:-8010}"
LOCAL_MODEL_ALIAS="${LOCAL_MODEL_ALIAS:-15961}"

tunnel_cmd() {
  printf 'ssh -N -o ExitOnForwardFailure=yes -o BatchMode=yes'
  printf ' -L %s:127.0.0.1:%s' "${REMOTE_VLLM_PORT}" "${REMOTE_VLLM_PORT}"
  printf ' -L %s:127.0.0.1:%s' "${REMOTE_MODEL_PORT}" "${REMOTE_MODEL_PORT}"
  printf ' -L %s:127.0.0.1:%s' "${LOCAL_VLLM_ALIAS}" "${REMOTE_VLLM_PORT}"
  printf ' -L %s:127.0.0.1:%s' "${LOCAL_MODEL_ALIAS}" "${REMOTE_MODEL_PORT}"
  printf ' %s' "${SSH_HOST}"
}

usage() {
  cat <<EOF
Usage: $(basename "$0") <start|stop|restart|status|check|discover>

  start    Launch ssh -N in a separate tmux session "${SESSION_NAME}"
  stop     Kill tmux session and tunnel
  restart  stop then start
  status   Show tmux session and listening local ports
  check    Health-check model service and vLLM through local aliases
  discover Print remote listening ports and running services on ${SSH_HOST}
EOF
}

discover_remote() {
  ssh -o BatchMode=yes "${SSH_HOST}" bash -s <<'REMOTE'
set -euo pipefail
echo "== Listening ports (vLLM / model service) =="
ss -ltn | grep -E ':(8000|8010|15959|15961)\s' || echo "(none on expected ports)"
echo
echo "== model_service / vllm processes =="
pgrep -af 'model_service|vllm serve' || echo "(none)"
echo
echo "== Remote health =="
for url in http://127.0.0.1:15959/status http://127.0.0.1:15961/status; do
  if curl -sf --connect-timeout 3 "$url" >/dev/null 2>&1; then
    echo "OK $url"
    curl -s "$url" | head -c 300
    echo
    break
  fi
done
for url in http://127.0.0.1:8000/health http://127.0.0.1:8010/health; do
  code="$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 3 "$url" || true)"
  if [[ "$code" == "200" ]]; then
    echo "OK $url (HTTP $code)"
    break
  fi
done
REMOTE
}

start_tunnel() {
  if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "Tunnel session already running: ${SESSION_NAME}"
    tmux list-panes -t "${SESSION_NAME}" -F '#{pane_current_command}' 2>/dev/null || true
    return 0
  fi

  local cmd
  cmd="$(tunnel_cmd)"
  # Dedicated session for ssh -N only; do not add to an existing eval/work session.
  tmux new-session -d -s "${SESSION_NAME}" -n ssh "${cmd}"
  sleep 1

  if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "FAIL: tmux session ${SESSION_NAME} did not start" >&2
    exit 1
  fi
  echo "Started ssh tunnel in separate tmux session: ${SESSION_NAME}"
  echo "Attach: tmux attach -t ${SESSION_NAME}"
  echo "Command: ${cmd}"
}

stop_tunnel() {
  if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    tmux kill-session -t "${SESSION_NAME}"
    echo "Stopped tmux session: ${SESSION_NAME}"
  else
    pkill -f "ssh -N.*${SSH_HOST}" 2>/dev/null || true
    echo "No tmux session ${SESSION_NAME}; killed stray ssh -N if any"
  fi
}

show_status() {
  if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "separate tmux session: ${SESSION_NAME} (running)"
    tmux list-windows -t "${SESSION_NAME}" -F '  window: #{window_name} (#{pane_current_command})' 2>/dev/null || true
  else
    echo "tmux session: ${SESSION_NAME} (not running)"
  fi
  echo
  echo "Local listeners:"
  ss -ltn 2>/dev/null | grep -E ":(8000|8010|15959|15961)\s" || echo "  (none)"
}

check_health() {
  local model_url="http://127.0.0.1:${LOCAL_MODEL_ALIAS}"
  local vllm_url="http://127.0.0.1:${LOCAL_VLLM_ALIAS}"

  echo "== Model service: ${model_url}/status =="
  if curl -sf --connect-timeout 5 "${model_url}/status"; then
    echo
    echo "OK"
  else
    echo "FAIL: model service not reachable via tunnel" >&2
    exit 1
  fi

  echo
  echo "== vLLM health: ${vllm_url}/health =="
  local code
  code="$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 5 "${vllm_url}/health" || true)"
  if [[ "${code}" != "200" ]]; then
    echo "FAIL: vLLM health returned HTTP ${code}" >&2
    exit 1
  fi
  echo "OK (HTTP ${code})"
}

main() {
  local action="${1:-}"
  case "${action}" in
    start) start_tunnel ;;
    stop) stop_tunnel ;;
    restart) stop_tunnel; start_tunnel ;;
    status) show_status ;;
    check) check_health ;;
    discover) discover_remote ;;
    *) usage; exit 1 ;;
  esac
}

main "${@:-}"
