#!/usr/bin/env bash
# Shared helpers for launch-dart-rollouter skill scripts.
set -euo pipefail

_LIB_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "${_LIB_SCRIPT}/.." && pwd)"
SCRIPTS_DIR="${SKILL_DIR}/scripts"
MANIFEST_DEFAULTS="${SKILL_DIR}/config/manifest.defaults.yaml"
MANIFEST_LOCAL="${SKILL_DIR}/config/manifest.local.yaml"

resolve_manifest_path() {
  if [[ -n "${DART_ROLLOUTER_MANIFEST:-}" ]]; then
    printf '%s' "${DART_ROLLOUTER_MANIFEST}"
    return
  fi
  if [[ ! -f "${MANIFEST_LOCAL}" ]]; then
    if [[ ! -f "${MANIFEST_DEFAULTS}" ]]; then
      echo "FAIL: missing manifest defaults: ${MANIFEST_DEFAULTS}" >&2
      exit 1
    fi
    cp "${MANIFEST_DEFAULTS}" "${MANIFEST_LOCAL}"
    echo "Created local manifest from defaults: ${MANIFEST_LOCAL}" >&2
  fi
  printf '%s' "${MANIFEST_LOCAL}"
}

MANIFEST="$(resolve_manifest_path)"

load_manifest() {
  MANIFEST="$(resolve_manifest_path)"
  if [[ ! -f "${MANIFEST}" ]]; then
    echo "FAIL: manifest not found: ${MANIFEST}" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  eval "$(python3 "${SCRIPTS_DIR}/resolve_config.py" \
    --manifest "${MANIFEST}" \
    --device "${DEVICE:-}" \
    --profile "${PROFILE:-}" \
    --format shell)"
  export DEVICE PROFILE SSH_HOST REPO_ROOT SESSION_NAME
  export PYTHON VENV_MODE REQUIREMENTS VLLM_PORT MODEL_PORT TENSOR_PARALLEL_SIZE SERVICE_CMD
  export REPO_ROOT SESSION_NAME VLLM_PORT MODEL_PORT MODEL_SERVICE_PORT="${MODEL_PORT}"
}

list_manifest() {
  python3 "${SCRIPTS_DIR}/resolve_config.py" --manifest "${MANIFEST}" --list
}

venv_python() {
  load_manifest
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    printf '%s' "${REPO_ROOT}/.venv/bin/python"
  else
    printf '%s' "${PYTHON}"
  fi
}

check_venv() {
  load_manifest
  local py
  py="$(venv_python)"

  echo "Repo:      ${REPO_ROOT}"
  echo "Device:    ${DEVICE}  Profile: ${PROFILE}"
  echo "Venv mode: ${VENV_MODE}"
  echo "Python:    ${py}"

  if [[ ! -x "${py}" ]]; then
    echo "FAIL: python not found at ${py}" >&2
    echo "Run: DEVICE=${DEVICE} PROFILE=${PROFILE} $(basename "${BASH_SOURCE[0]}") install" >&2
    exit 1
  fi

  "${py}" - <<'PY'
import sys

def check(name, fn):
    try:
        value = fn()
        print(f"OK  {name}: {value}")
        return True
    except Exception as exc:
        print(f"FAIL {name}: {exc}")
        return False

ok = True
ok &= check("python", lambda: sys.executable)
ok &= check("vllm", lambda: __import__("vllm").__version__)
ok &= check("torch", lambda: f"{__import__('torch').__version__} cuda={__import__('torch').cuda.is_available()} gpus={__import__('torch').cuda.device_count()}")
for pkg in ("hydra", "omegaconf", "pynvml"):
    ok &= check(pkg, lambda p=pkg: f"{p} ok" if __import__(p) else f"{p} missing")
raise SystemExit(0 if ok else 1)
PY
}

install_venv() {
  load_manifest
  local req_file="${REQUIREMENTS}"
  if [[ -z "${req_file}" ]]; then
    req_file="dart_rollouter/requirements_vllm_complementary.txt"
  fi
  cd "${REPO_ROOT}"

  if [[ "${VENV_MODE}" == "bundled" ]]; then
    if [[ -x ".venv/bin/python" ]] && .venv/bin/python -c "import vllm" 2>/dev/null; then
      echo "Bundled venv already has vllm — skipping recreate."
      check_venv
      return 0
    fi
    echo "Creating bundled venv with full inference stack..."
    uv venv --python "${PYTHON}" .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    uv pip install -r "${req_file}"
    check_venv
    return 0
  fi

  echo "Creating complementary venv (system-site-packages)..."
  uv venv --system-site-packages --python "${PYTHON}" .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install --no-deps -r "${req_file}"
  check_venv
}
