#!/usr/bin/env bash
# Launch OSWorld uitars eval in a dedicated tmux session (survives terminal detach).
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-osworld-eval}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

usage() {
  cat <<EOF
Usage: $(basename "$0") <start|stop|restart|status|attach> [eval command...]

  start    Run eval in detached tmux session "${SESSION_NAME}"
           Command via EVAL_CMD env or arguments after "start" (joined with spaces).
  stop     Kill tmux session "${SESSION_NAME}"
  restart  stop then start (requires EVAL_CMD or args as for start)
  status   Show whether eval session is running
  attach   Attach to session (Ctrl+B then D to detach)

Examples:
  EVAL_CMD='source ${REPO_ROOT}/.venv/bin/activate && cd ${REPO_ROOT}/GUI-Docker-Env && PYTHONPATH=. python uitars_run/run_uitars.py ...' \\
    $(basename "$0") start

  $(basename "$0") start 'source .venv/bin/activate && cd GUI-Docker-Env && PYTHONPATH=. python uitars_run/run_uitars.py --max-workers 1 ...'
EOF
}

resolve_eval_cmd() {
  local cmd="${EVAL_CMD:-}"
  if [[ $# -gt 0 ]]; then
    cmd="$*"
  fi
  if [[ -z "${cmd}" ]]; then
    echo "FAIL: set EVAL_CMD or pass the eval shell command after 'start'" >&2
    usage >&2
    exit 1
  fi
  printf '%s' "${cmd}"
}

start_eval() {
  local eval_cmd
  eval_cmd="$(resolve_eval_cmd "$@")"

  if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    if [[ "${FORCE:-}" == "1" ]]; then
      tmux kill-session -t "${SESSION_NAME}"
    else
      echo "FAIL: tmux session ${SESSION_NAME} already exists (eval may still be running)" >&2
      echo "  status: $(basename "$0") status" >&2
      echo "  attach: $(basename "$0") attach" >&2
      echo "  stop:   $(basename "$0") stop" >&2
      echo "  retry:  FORCE=1 $(basename "$0") start ..." >&2
      exit 1
    fi
  fi

  # Dedicated session for uitars_run only; do not reuse dart-rollouter-tunnel or other sessions.
  tmux new-session -d -s "${SESSION_NAME}" -n eval "bash -lc $(printf '%q' "${eval_cmd}")"
  sleep 1

  if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "FAIL: tmux session ${SESSION_NAME} did not start" >&2
    exit 1
  fi

  echo "Started eval in tmux session: ${SESSION_NAME}"
  echo "Attach:  tmux attach -t ${SESSION_NAME}"
  echo "Status:  $(basename "$0") status"
  echo "Repo:    ${REPO_ROOT}"
}

stop_eval() {
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

attach_eval() {
  if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "FAIL: no session ${SESSION_NAME}. Start eval with: $(basename "$0") start" >&2
    exit 1
  fi
  exec tmux attach -t "${SESSION_NAME}"
}

cmd="${1:-}"
shift || true

case "${cmd}" in
  start) start_eval "$@" ;;
  stop) stop_eval ;;
  restart)
    stop_eval
    start_eval "$@"
    ;;
  status) show_status ;;
  attach) attach_eval ;;
  -h|--help|help|"") usage ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    usage >&2
    exit 1
    ;;
esac
