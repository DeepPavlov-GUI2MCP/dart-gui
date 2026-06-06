#!/usr/bin/env bash
# Back-compat wrapper: start model_service in tmux (honors DEVICE / PROFILE).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/scripts/run_service.sh" start "$@"
