from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class MlflowSettings:
    enabled: bool = False
    experiment_name: str = "dart-uitars-sft"
    run_name: str | None = None


def resolve_mlflow_tracking_uri() -> str | None:
    host = os.environ.get("MLFLOW_HOST", "127.0.0.1").strip()
    port = os.environ.get("MLFLOW_PORT", "5000").strip()
    if not host or not port:
        return None
    if os.environ.get("MLFLOW_TRACKING_URI", "").strip():
        return os.environ["MLFLOW_TRACKING_URI"].strip()
    return f"http://{host}:{port}"


def configure_mlflow(settings: MlflowSettings) -> bool:
    if not settings.enabled:
        return False
    tracking_uri = resolve_mlflow_tracking_uri()
    if not tracking_uri:
        raise ValueError("mlflow.enabled=true but MLFLOW tracking URI could not be resolved")
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
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
