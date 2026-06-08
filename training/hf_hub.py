from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypeVar

REPO_ROOT = Path(__file__).resolve().parents[1]

T = TypeVar("T")


def load_repo_dotenv() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_path, override=False)


def resolve_hf_token(config_token: str | None = None) -> str | None:
    if config_token and config_token.strip():
        return config_token.strip()
    for name in ("HF_TOKEN", "HUGGINGFACE_API_KEY", "HF_API_KEY", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(name, "").strip()
        if token:
            return token
    return None


def apply_hf_home_override() -> None:
    dart_hf = os.environ.get("DART_HF_HOME", "").strip() or os.environ.get("SMIT_HF_HOME", "").strip()
    if dart_hf:
        os.environ["HF_HOME"] = str(Path(dart_hf).expanduser().resolve())


def configure_hf_hub(*, config_token: str | None = None) -> str | None:
    load_repo_dotenv()
    apply_hf_home_override()
    token = resolve_hf_token(config_token)
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACE_API_KEY"] = token
        os.environ["HF_API_KEY"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    return token


def from_pretrained_kwargs(
    *,
    trust_remote_code: bool = True,
    token: str | None = None,
    prefer_local: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if extra:
        kwargs.update(extra)
    resolved_token = resolve_hf_token(token)
    if resolved_token:
        kwargs["token"] = resolved_token
    if prefer_local:
        kwargs["local_files_only"] = True
    return kwargs


def load_pretrained(
    model_cls: type[T],
    model_id: str,
    *,
    trust_remote_code: bool = True,
    token: str | None = None,
    prefer_local: bool = True,
    **extra: Any,
) -> T:
    local_kwargs = from_pretrained_kwargs(
        trust_remote_code=trust_remote_code,
        token=token,
        prefer_local=True,
        extra=extra,
    )
    if prefer_local:
        try:
            return model_cls.from_pretrained(model_id, **local_kwargs)
        except (OSError, ValueError, EnvironmentError):
            pass
    remote_kwargs = from_pretrained_kwargs(
        trust_remote_code=trust_remote_code,
        token=token,
        prefer_local=False,
        extra=extra,
    )
    return model_cls.from_pretrained(model_id, **remote_kwargs)
