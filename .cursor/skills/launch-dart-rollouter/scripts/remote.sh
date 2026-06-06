#!/usr/bin/env bash
# Run run_service.sh on a remote GPU device via SSH (from CPU machine or laptop).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") <start|stop|restart|status|attach|check|check-venv|install|list> [--device NAME] [--profile NAME]

Runs the launch-dart-rollouter skill on a remote GPU host defined in manifest.yaml.

  list      Show devices and profiles
  install   Create remote .venv (runs install on target host)
  start     Start model service in remote tmux session
  check     SSH + curl health on remote host

Environment:
  DEVICE, PROFILE   Select target from config/manifest.yaml
  FORCE=1           Replace existing remote tmux session on start

Examples:
  PROFILE=holo3-2gpu $(basename "$0") start
  $(basename "$0") --device rollouter-gpu2 --profile uitars-1.5 check
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --device|-d)
        DEVICE="$2"
        shift 2
        ;;
      --profile|-p)
        PROFILE="$2"
        shift 2
        ;;
      *)
        REMOTE_CMD="$1"
        shift
        REMOTE_ARGS=("$@")
        return 0
        ;;
    esac
  done
  REMOTE_CMD=""
  REMOTE_ARGS=()
}

remote_run() {
  local action="$1"
  shift
  load_manifest

  if [[ -z "${SSH_HOST:-}" ]]; then
    echo "Device ${DEVICE} has no ssh_host — run scripts/run_service.sh locally instead." >&2
    exit 1
  fi

  local remote_script="${REPO_ROOT}/.cursor/skills/launch-dart-rollouter/scripts/run_service.sh"

  echo "Remote: ${SSH_HOST}  Device: ${DEVICE}  Profile: ${PROFILE}"
  local remote_cmd="DEVICE=${DEVICE} PROFILE=${PROFILE} REPO_ROOT=${REPO_ROOT} DART_ROLLOUTER_MANIFEST=${REPO_ROOT}/.cursor/skills/launch-dart-rollouter/config/manifest.yaml"
  if [[ -n "${FORCE:-}" ]]; then
    remote_cmd+=" FORCE=1"
  fi
  if [[ -n "${SERVICE_CMD:-}" ]]; then
    remote_cmd+=" SERVICE_CMD=$(printf '%q' "${SERVICE_CMD}")"
  fi
  remote_cmd+=" bash ${remote_script} ${action}"
  if [[ $# -gt 0 ]]; then
    remote_cmd+=" $(printf '%q ' "$@")"
  fi
  ssh -o BatchMode=yes "${SSH_HOST}" "${remote_cmd}"
}

REMOTE_CMD=""
REMOTE_ARGS=()
parse_args "$@"

case "${REMOTE_CMD:-}" in
  list) list_manifest ;;
  install|start|stop|restart|status|attach|check|check-venv)
    remote_run "${REMOTE_CMD}" "${REMOTE_ARGS[@]}"
    ;;
  -h|--help|help|"") usage ;;
  *)
    echo "Unknown command: ${REMOTE_CMD:-}" >&2
    usage >&2
    exit 1
    ;;
esac
