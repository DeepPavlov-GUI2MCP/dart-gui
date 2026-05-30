#!/usr/bin/env bash
# Report running emulators and free token quota before asking the user for parallel count.
set -euo pipefail

DESKTOP_URL="${DESKTOP_URL:-http://127.0.0.1:50003}"
DESKTOP_TOKEN="${DESKTOP_TOKEN:-dart}"

if ! curl -sf "${DESKTOP_URL}/ping" >/dev/null; then
  echo "FAIL: desktop server not reachable at ${DESKTOP_URL}" >&2
  exit 1
fi

status_json="$(curl -sf "${DESKTOP_URL}/status")"
emulators_json="$(curl -sf "${DESKTOP_URL}/emulators?token=${DESKTOP_TOKEN}")"

DESKTOP_TOKEN="${DESKTOP_TOKEN}" STATUS_JSON="${status_json}" EMULATORS_JSON="${emulators_json}" python3 - <<'PY'
import json, os, sys

token = os.environ["DESKTOP_TOKEN"]
status = json.loads(os.environ["STATUS_JSON"])
emulators = json.loads(os.environ["EMULATORS_JSON"])

info = next((t for t in status.get("tokens", []) if t.get("token") == token), None)
if info is None:
    print(f"FAIL: token '{token}' not found in /status", file=sys.stderr)
    sys.exit(1)

limit = int(info.get("limit", 0))
current = int(info.get("current", 0))
pending = int(info.get("pending", 0))
available = int(info.get("available", max(0, limit - current - pending)))
running_total = int(status.get("total_emulators", len(emulators)))

lines = [
    "== Emulator quota check ==",
    f"token:              {token}",
    f"token limit:        {limit}",
    f"in use (current):   {current}",
    f"starting (pending): {pending}",
    f"free slots:         {available}",
    f"running emulators:  {running_total}",
]

if emulators:
    lines.append("")
    lines.append("Active emulators:")
    for emu in emulators:
        eid = emu.get("emulator_id", "?")
        mins = emu.get("duration_minutes", "?")
        vnc = emu.get("vnc_port", "?")
        lines.append(f"  - {eid}  vnc={vnc}  uptime_min={mins}")
else:
    lines.append("")
    lines.append("Active emulators: none")

lines.append("")
if running_total > 0 or current > 0:
    lines.append(
        "NOTE: Emulators are already running. If the requested parallel count "
        f"is greater than free slots ({available}), existing emulators must be "
        "stopped before the eval can start."
    )
    lines.append(
        "      Confirm with the user before calling /stop_all_emulators."
    )
else:
    lines.append("NOTE: No emulators running. Free slots are ready for a new eval.")

print("\n".join(lines))

# Machine-readable summary for the agent (last line)
summary = {
    "token": token,
    "limit": limit,
    "current": current,
    "pending": pending,
    "available": available,
    "running_emulators": running_total,
    "emulator_ids": [e.get("emulator_id") for e in emulators],
}
print("SUMMARY_JSON=" + json.dumps(summary))
PY
