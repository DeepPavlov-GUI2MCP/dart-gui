#!/usr/bin/env bash
# Preflight checks before OSWorld UI-TARS eval via dart-rollouter vLLM.
set -euo pipefail

VLLM_BASE="${VLLM_BASE:-http://127.0.0.1:8010}"
DESKTOP_URL="${DESKTOP_URL:-http://127.0.0.1:50003}"
MODEL_SERVICE_URL="${MODEL_SERVICE_URL:-http://127.0.0.1:15961}"
REQUESTED_MODEL="${REQUESTED_MODEL:-ByteDance-Seed/UI-TARS-1.5-7B}"
OPENAI_API_KEY="${OPENAI_API_KEY:-empty}"
EMULATOR_COUNT="${EMULATOR_COUNT:-1}"
DESKTOP_TOKEN="${DESKTOP_TOKEN:-dart}"

echo "== OSWorld eval preflight =="
echo "Requested emulators: ${EMULATOR_COUNT}"

# 1) Desktop server + token quota
echo "[1/5] Desktop server: ${DESKTOP_URL}/ping"
if ! curl -sf "${DESKTOP_URL}/ping" >/dev/null; then
  echo "FAIL: desktop server not reachable at ${DESKTOP_URL}" >&2
  exit 1
fi
echo "OK"

status_json="$(curl -sf "${DESKTOP_URL}/status")"
quota_ok="$(STATUS_JSON="${status_json}" DESKTOP_URL="${DESKTOP_URL}" DESKTOP_TOKEN="${DESKTOP_TOKEN}" EMULATOR_COUNT="${EMULATOR_COUNT}" python3 - <<'PY'
import json, os, sys
status = json.loads(os.environ["STATUS_JSON"])
token = os.environ["DESKTOP_TOKEN"]
want = int(os.environ["EMULATOR_COUNT"])
desktop_url = os.environ["DESKTOP_URL"]
info = next((t for t in status.get("tokens", []) if t.get("token") == token), None)
if info is None:
    print(f"FAIL: token '{token}' not found in /status", file=sys.stderr)
    sys.exit(1)
limit = int(info.get("limit", 0))
current = int(info.get("current", 0))
available = int(info.get("available", max(0, limit - current)))
running = int(status.get("total_emulators", 0))
print(f"token={token} limit={limit} current={current} available={available} running={running}")
if want > limit:
    print(f"FAIL: requested {want} emulators but token limit is {limit}", file=sys.stderr)
    print(
        f"Hint: curl -X POST {desktop_url}/set_token_limit "
        f"-H 'Content-Type: application/json' "
        f"-d '{{\"token\":\"{token}\",\"limit\":{want}}}'",
        file=sys.stderr,
    )
    sys.exit(1)
if available < want and current > 0:
    print(f"WARN: only {available} slot(s) free now; {current} emulator(s) already running", file=sys.stderr)
PY
)" || {
  echo "${quota_ok}" >&2
  exit 1
}
echo "Quota: ${quota_ok}"

# 2) dart-rollouter model service (optional but informative)
echo "[2/5] Model service: ${MODEL_SERVICE_URL}/status"
if curl -sf "${MODEL_SERVICE_URL}/status" >/dev/null 2>&1; then
  curl -s "${MODEL_SERVICE_URL}/status" | python3 -m json.tool 2>/dev/null || curl -s "${MODEL_SERVICE_URL}/status"
  echo "OK"
else
  echo "WARN: model service not reachable (vLLM may still work via SSH tunnel)"
fi

# 3) vLLM health + list models
echo "[3/5] vLLM health: ${VLLM_BASE}/health"
code="$(curl -s -o /dev/null -w '%{http_code}' "${VLLM_BASE}/health" || true)"
if [[ "${code}" != "200" ]]; then
  echo "FAIL: vLLM health returned HTTP ${code} at ${VLLM_BASE}" >&2
  echo "Hint: start SSH tunnel: ssh -N dart-rollouter" >&2
  exit 1
fi
echo "OK"

models_json="$(curl -sf "${VLLM_BASE}/v1/models")"
model_id="$(REQUESTED_MODEL="${REQUESTED_MODEL}" MODELS_JSON="${models_json}" python3 - <<'PY'
import json, os, sys

requested = os.environ["REQUESTED_MODEL"]
payload = json.loads(os.environ["MODELS_JSON"])
ids = [m["id"] for m in payload.get("data", [])]
if not ids:
    sys.exit(0)

aliases = {
    requested,
    "ui_tars_1.5",
    "UI-TARS-1.5",
    "UI-TARS-1.5-7B",
    "ByteDance-Seed/UI-TARS-1.5-7B",
}
for mid in ids:
    if mid in aliases:
        print(mid)
        sys.exit(0)
for mid in ids:
    low = mid.lower()
    if "ui-tars" in low or "bytedance-seed" in low:
        print(mid)
        sys.exit(0)
if len(ids) == 1:
    print(ids[0])
PY
)"

if [[ -z "${model_id}" ]]; then
  echo "FAIL: could not resolve a vLLM model id for ${REQUESTED_MODEL}" >&2
  echo "Available models:" >&2
  echo "${models_json}" | python3 -m json.tool >&2 || echo "${models_json}" >&2
  exit 1
fi

echo "Available vLLM models:"
echo "${models_json}" | python3 -m json.tool 2>/dev/null || echo "${models_json}"
echo "Resolved model id: ${model_id}"
export RESOLVED_VLLM_MODEL_ID="${model_id}"

# 4) Sanity chat/completions request
echo "[4/5] Sanity request: POST ${VLLM_BASE}/v1/chat/completions (model=${model_id})"
sanity_payload="$(MODEL_ID="${model_id}" python3 - <<'PY'
import json, os
print(json.dumps({
    "model": os.environ["MODEL_ID"],
    "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
    "max_tokens": 8,
    "temperature": 0,
}))
PY
)"

sanity_code="$(curl -s -o /tmp/eval_osworld_preflight.json -w '%{http_code}' \
  -H "Authorization: Bearer ${OPENAI_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "${sanity_payload}" \
  "${VLLM_BASE}/v1/chat/completions" || true)"

if [[ "${sanity_code}" != "200" ]]; then
  echo "FAIL: sanity chat/completions returned HTTP ${sanity_code}" >&2
  cat /tmp/eval_osworld_preflight.json >&2 || true
  exit 1
fi

echo "Sanity response:"
python3 -m json.tool /tmp/eval_osworld_preflight.json 2>/dev/null | head -20 || cat /tmp/eval_osworld_preflight.json

# 5) Parallel readiness summary
echo "[5/5] Parallel readiness"
echo "  max-workers should be: ${EMULATOR_COUNT}"
echo "  token limit should be:  ${EMULATOR_COUNT} (set via /set_token_limit if needed)"
echo "  keep SSH tunnel alive for entire run"
echo
echo "PREFLIGHT OK — model=${model_id} emulators=${EMULATOR_COUNT}"
