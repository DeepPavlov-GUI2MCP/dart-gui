from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_HF_HOME_REL = "data/.cache/huggingface"


def repo_root() -> Path:
    return _REPO_ROOT


def resolve_hf_home() -> str:
    override = os.environ.get("DART_HF_HOME", "").strip() or os.environ.get("SMIT_HF_HOME", "").strip()
    if override:
        return str(Path(override).expanduser().resolve())
    return str((_REPO_ROOT / DEFAULT_HF_HOME_REL).resolve())


def hf_hub_env() -> dict[str, str]:
    return {"HF_HOME": resolve_hf_home()}


def apply_hf_hub_env() -> str:
    hf_home = resolve_hf_home()
    os.environ["HF_HOME"] = hf_home
    return hf_home


def require_hf_token() -> str | None:
    for name in ("HF_TOKEN", "HUGGINGFACE_API_KEY", "HF_API_KEY"):
        token = os.environ.get(name, "").strip()
        if token:
            return token
    return None


def forward_hf_token_env() -> dict[str, str]:
    token = require_hf_token()
    if token:
        return {"HF_TOKEN": token, "HUGGINGFACE_API_KEY": token}
    return {}


def warn_missing_hf_token() -> None:
    if require_hf_token() is None:
        print("warning: HF_TOKEN not set; workers may fail to download uncached models", file=sys.stderr)
