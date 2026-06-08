#!/usr/bin/env bash
# Stop tracked emulators and remove orphaned OSWorld Docker containers on each desktop host.
set -euo pipefail

DESKTOP_TOKEN="${DESKTOP_TOKEN:-dart}"
DESKTOP_HOSTS="${DESKTOP_HOSTS:-${DESKTOP_URL:-http://127.0.0.1:50003}}"
OSWORLD_IMAGE="${OSWORLD_IMAGE:-happysixd/osworld-docker}"
STOP_EMULATORS="${STOP_EMULATORS:-1}"

usage() {
  cat <<EOF
Usage: $(basename "$0")

Stops emulators and removes stale ${OSWORLD_IMAGE} containers on every host in DESKTOP_HOSTS.

Environment:
  DESKTOP_HOSTS   Comma-separated desktop server base URLs, or name=url pairs
                  (default: DESKTOP_URL or http://127.0.0.1:50003)
  DESKTOP_URL     Single-host fallback when DESKTOP_HOSTS is unset
  DESKTOP_TOKEN   Bearer token (default: dart)
  STOP_EMULATORS  1 to call /stop_all_emulators first (default: 1)
  OSWORLD_IMAGE   Docker image filter for /remove_all_containers (default: happysixd/osworld-docker)

Example:
  DESKTOP_HOSTS='http://94.139.251.45:50003,http://127.0.0.1:50003' \\
    bash $(basename "$0")
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

parse_hosts() {
  local raw="$1"
  local item url name
  IFS=',' read -ra items <<< "${raw}"
  for item in "${items[@]}"; do
    item="$(echo "${item}" | xargs)"
    [[ -z "${item}" ]] && continue
    if [[ "${item}" == *"="* ]]; then
      name="${item%%=*}"
      url="${item#*=}"
      name="$(echo "${name}" | xargs)"
      url="$(echo "${url}" | xargs)"
    else
      url="${item}"
      name="${url}"
    fi
    url="${url%/}"
    printf '%s\t%s\n' "${name}" "${url}"
  done
}

cleanup_host() {
  local name="$1"
  local base="$2"
  local auth_header=()
  if [[ -n "${DESKTOP_TOKEN}" ]]; then
    auth_header=(-H "Authorization: Bearer ${DESKTOP_TOKEN}")
  fi

  echo "== ${name} (${base}) =="

  if ! curl -sf "${base}/ping" >/dev/null; then
    echo "FAIL: desktop server not reachable at ${base}" >&2
    return 1
  fi

  if [[ "${STOP_EMULATORS}" == "1" ]]; then
    echo "  stop_all_emulators..."
    curl -sf -X POST "${base}/stop_all_emulators" "${auth_header[@]}" >/dev/null || true
    sleep 2
  fi

  echo "  remove_all_containers (${OSWORLD_IMAGE})..."
  remove_json="$(curl -sf -X POST "${base}/remove_all_containers" \
    "${auth_header[@]}" \
    -H "Content-Type: application/json" \
    -d "{\"image\":\"${OSWORLD_IMAGE}\",\"min_age\":0,\"dry_run\":false}" || true)"
  if [[ -n "${remove_json}" ]]; then
    echo "  ${remove_json}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"  removed={d.get('removed_count',0)} failed={d.get('failed_count',0)} msg={d.get('message','')}\")" 2>/dev/null || echo "  ${remove_json}"
  else
    echo "  WARN: remove_all_containers request failed"
  fi

  case "${base}" in
    http://127.0.0.1:*|http://localhost:*)
      if command -v docker >/dev/null 2>&1; then
        docker container prune -f >/dev/null 2>&1 || true
      fi
      ;;
  esac

  status_json="$(curl -sf "${base}/status")"
  echo "  $(STATUS_JSON="${status_json}" DESKTOP_TOKEN="${DESKTOP_TOKEN}" python3 - <<'PY'
import json, os
status = json.loads(os.environ["STATUS_JSON"])
token = os.environ["DESKTOP_TOKEN"]
info = next((t for t in status.get("tokens", []) if t.get("token") == token), None)
if info is None:
    print("status: token missing")
else:
    print(
        f"status: emulators={status.get('total_emulators', 0)} "
        f"current={info.get('current', 0)} pending={info.get('pending', 0)} "
        f"limit={info.get('limit', 0)}"
    )
PY
)"
}

echo "== Desktop host cleanup =="
echo "Hosts: ${DESKTOP_HOSTS}"

fail=0
while IFS=$'\t' read -r name url; do
  [[ -z "${url}" ]] && continue
  if ! cleanup_host "${name}" "${url}"; then
    fail=1
  fi
done < <(parse_hosts "${DESKTOP_HOSTS}")

if [[ "${fail}" -ne 0 ]]; then
  echo "FAIL: one or more desktop hosts could not be cleaned" >&2
  exit 1
fi

echo "Desktop host cleanup OK"
