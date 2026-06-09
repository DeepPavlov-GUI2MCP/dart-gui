from __future__ import annotations

import os
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse


@dataclass(frozen=True)
class MlflowSettings:
    enabled: bool = False
    experiment_name: str = "dart-uitars-sft"
    run_name: str | None = None


def resolve_mlflow_repo_root() -> Path:
    for key in ("DART_REPO_ROOT", "REPO_ROOT"):
        value = os.environ.get(key, "").strip()
        if value:
            return Path(value).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def _as_file_uri(path: Path) -> str:
    return f"file://{path.resolve()}"


def _normalize_file_uri(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("file://"):
        return stripped
    return _as_file_uri(Path(stripped).expanduser())


def resolve_mlflow_store(*, output_dir: Path | None = None) -> tuple[str, str]:
    backend_override = os.environ.get("MLFLOW_BACKEND_STORE_URI", "").strip()
    if backend_override:
        backend_uri = _normalize_file_uri(backend_override)
    else:
        backend_uri = _as_file_uri(resolve_mlflow_repo_root() / "outputs" / "sft" / "mlruns")

    artifact_override = os.environ.get("MLFLOW_ARTIFACT_ROOT", "").strip()
    if artifact_override:
        artifact_root = _normalize_file_uri(artifact_override)
    elif output_dir is not None:
        artifact_root = _as_file_uri(output_dir / "mlartifacts")
    else:
        artifact_root = _as_file_uri(resolve_mlflow_repo_root() / "outputs" / "sft" / "mlartifacts")
    return backend_uri, artifact_root


def prepare_mlflow_file_store(*, output_dir: Path | None = None) -> tuple[str, str]:
    backend_uri, artifact_root = resolve_mlflow_store(output_dir=output_dir)
    Path(backend_uri.removeprefix("file://")).mkdir(parents=True, exist_ok=True)
    Path(artifact_root.removeprefix("file://")).mkdir(parents=True, exist_ok=True)
    os.environ["MLFLOW_BACKEND_STORE_URI"] = backend_uri
    os.environ["MLFLOW_ARTIFACT_ROOT"] = artifact_root
    return backend_uri, artifact_root


def resolve_mlflow_connect_host() -> str:
    explicit = os.environ.get("MLFLOW_HOST", "").strip()
    if explicit:
        return explicit
    if int(os.environ.get("DART_JOB_NUM_NODES", "1")) <= 1:
        return "127.0.0.1"
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        return "127.0.0.1"
    master = os.environ.get("MASTER_ADDR", "").strip()
    if master:
        return master
    return "127.0.0.1"


def resolve_mlflow_port() -> int:
    raw = os.environ.get("MLFLOW_PORT", "5000").strip()
    if not raw:
        return 5000
    return int(raw)


def resolve_mlflow_tracking_uri() -> str:
    explicit = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if explicit and not explicit.startswith("file://"):
        return explicit
    host = resolve_mlflow_connect_host()
    port = resolve_mlflow_port()
    return f"http://{host}:{port}"


def _parse_http_uri(uri: str) -> tuple[str, int]:
    parsed = urlparse(uri)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or resolve_mlflow_port()
    return host, port


def is_mlflow_server_reachable(uri: str, *, timeout_s: float = 1.0) -> bool:
    host, port = _parse_http_uri(uri)
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def wait_for_mlflow_server(uri: str, *, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if is_mlflow_server_reachable(uri):
            return
        time.sleep(0.5)
    raise RuntimeError(f"MLflow server not reachable at {uri} after {timeout_s:.0f}s")


def resolve_mlflow_tmux_session() -> str:
    return os.environ.get("MLFLOW_TMUX_SESSION", "dart-sft-mlflow").strip() or "dart-sft-mlflow"


def tmux_has_session(session: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", session],
        check=False,
        capture_output=True,
    ).returncode == 0


def kill_listeners_on_port(port: int) -> None:
    subprocess.run(
        ["bash", "-lc", f"fuser -k {port}/tcp 2>/dev/null || true"],
        check=False,
    )


def stop_mlflow_tmux_server(*, port: int | None = None) -> None:
    port = resolve_mlflow_port() if port is None else port
    session = resolve_mlflow_tmux_session()
    if tmux_has_session(session):
        subprocess.run(["tmux", "kill-session", "-t", session], check=False)
    kill_listeners_on_port(port)


def _presubmit_mlflow_python() -> Path:
    repo = resolve_mlflow_repo_root()
    venv_python = repo / ".venv" / "bin" / "python"
    if venv_python.is_file():
        return venv_python
    return Path(sys.executable)


def _build_mlflow_server_shell_command(
    *,
    backend_uri: str,
    artifact_root: str,
    host: str,
    port: int,
    python: Path,
    repo_root: Path,
) -> str:
    return (
        f"cd {shlex.quote(str(repo_root))} && "
        f"export MLFLOW_ALLOW_FILE_STORE=true && "
        f"exec {shlex.quote(str(python))} -m mlflow server "
        f"--host {shlex.quote(host)} "
        f"--port {port} "
        f"--backend-store-uri {shlex.quote(backend_uri)} "
        f"--default-artifact-root {shlex.quote(_artifact_root_for_cli(artifact_root))}"
    )


def start_mlflow_server_tmux(
    *,
    backend_uri: str,
    artifact_root: str,
    host: str = "127.0.0.1",
    port: int | None = None,
) -> str:
    port = resolve_mlflow_port() if port is None else port
    session = resolve_mlflow_tmux_session()
    if tmux_has_session(session):
        raise RuntimeError(f"tmux session already exists: {session}")
    repo_root = resolve_mlflow_repo_root()
    python = _presubmit_mlflow_python()
    server_cmd = _build_mlflow_server_shell_command(
        backend_uri=backend_uri,
        artifact_root=artifact_root,
        host=host,
        port=port,
        python=python,
        repo_root=repo_root,
    )
    tmux_cmd = [
        "tmux",
        "new-session",
        "-d",
        "-s",
        session,
        "-n",
        "mlflow",
        f"bash -lc {shlex.quote(server_cmd)}",
    ]
    result = subprocess.run(tmux_cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"failed to start tmux session {session}: {detail}")
    if not tmux_has_session(session):
        raise RuntimeError(f"tmux session did not start: {session}")
    return session


def ensure_presubmit_mlflow_server(
    settings: MlflowSettings,
    *,
    output_dir: Path | None = None,
) -> str:
    if not settings.enabled:
        return os.environ.get("MLFLOW_TRACKING_URI", "").strip()

    backend_uri, artifact_root = prepare_mlflow_file_store(output_dir=output_dir)
    host = os.environ.get("MLFLOW_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = resolve_mlflow_port()
    tracking_uri = f"http://{host}:{port}"

    stop_mlflow_tmux_server(port=port)
    session = start_mlflow_server_tmux(
        backend_uri=backend_uri,
        artifact_root=artifact_root,
        host=host,
        port=port,
    )
    try:
        wait_for_mlflow_server(tracking_uri, timeout_s=90.0)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{exc}. Check tmux session {session!r}: tmux attach -t {session}"
        ) from exc

    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ.setdefault("MLFLOW_HOST", host)
    os.environ.setdefault("MLFLOW_PORT", str(port))
    return tracking_uri


def _artifact_root_for_cli(artifact_root: str) -> str:
    if artifact_root.startswith("file://"):
        return artifact_root.removeprefix("file://")
    return artifact_root


def configure_mlflow_file_tracking(
    *,
    output_dir: Path | None = None,
) -> str:
    explicit = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if explicit.startswith("file://"):
        prepare_mlflow_file_store(output_dir=output_dir)
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        return explicit

    backend_uri, _artifact_root = prepare_mlflow_file_store(output_dir=output_dir)
    os.environ["MLFLOW_TRACKING_URI"] = backend_uri
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    return backend_uri


def configure_mlflow(
    settings: MlflowSettings,
    *,
    output_dir: Path | None = None,
) -> bool:
    if not settings.enabled:
        return False

    explicit = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if explicit.startswith("http://") or explicit.startswith("https://"):
        os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", settings.experiment_name)
        if settings.run_name:
            os.environ.setdefault("MLFLOW_RUN_NAME", settings.run_name)
        return True

    tracking_uri = configure_mlflow_file_tracking(output_dir=output_dir)
    if not tracking_uri:
        raise ValueError("mlflow.enabled=true but MLFLOW tracking URI could not be resolved")
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", settings.experiment_name)
    if settings.run_name:
        os.environ.setdefault("MLFLOW_RUN_NAME", settings.run_name)
    return True


def _mlflow_metric_name(split: str, metric_key: str) -> str:
    if metric_key in {"mean_token_accuracy", "token_accuracy"}:
        return f"{split}/token_accuracy"
    return f"{split}/{metric_key}"


def prefix_trainer_logs_for_mlflow(logs: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    is_eval = any(key.startswith("eval_") for key in logs)
    for key, value in logs.items():
        if not isinstance(value, (int, float)):
            continue
        if key.startswith("eval_"):
            metrics[_mlflow_metric_name("val", key[5:])] = value
        elif key.startswith("train_"):
            metrics[_mlflow_metric_name("train", key[6:])] = value
        elif is_eval and key == "epoch":
            metrics["val/epoch"] = value
        else:
            metrics[_mlflow_metric_name("train", key)] = value
    return metrics


def add_final_summary_val_metrics(
    prefixed: dict[str, float],
    logs: Mapping[str, Any],
    *,
    last_val_mean_token_accuracy: float | None,
) -> dict[str, float]:
    if last_val_mean_token_accuracy is None:
        return prefixed
    if "train_loss" not in logs or "train_runtime" not in logs:
        return prefixed
    updated = dict(prefixed)
    updated["val/mean_token_accuracy"] = last_val_mean_token_accuracy
    return updated


def build_mlflow_callback():
    from transformers.integrations import MLflowCallback

    class PrefixedMlflowCallback(MLflowCallback):
        def __init__(self):
            super().__init__()
            self._last_val_mean_token_accuracy: float | None = None

        def on_log(self, args, state, control, logs, model=None, **kwargs):
            if logs is None:
                return
            if isinstance(logs.get("eval_mean_token_accuracy"), (int, float)):
                self._last_val_mean_token_accuracy = float(logs["eval_mean_token_accuracy"])
            prefixed = prefix_trainer_logs_for_mlflow(logs)
            prefixed = add_final_summary_val_metrics(
                prefixed,
                logs,
                last_val_mean_token_accuracy=self._last_val_mean_token_accuracy,
            )
            super().on_log(
                args,
                state,
                control,
                prefixed,
                model=model,
                **kwargs,
            )

    return PrefixedMlflowCallback()
