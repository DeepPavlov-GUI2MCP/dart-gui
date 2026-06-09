from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def import_training_module(name: str):
    sys.path.insert(0, str(REPO_ROOT / "training"))
    try:
        return __import__(name)
    finally:
        sys.path.pop(0)


def test_prefix_trainer_logs_for_mlflow_maps_train_metrics():
    mlflow_utils = import_training_module("mlflow_utils")
    metrics = mlflow_utils.prefix_trainer_logs_for_mlflow(
        {
            "loss": 0.5,
            "learning_rate": 1e-4,
            "epoch": 1.0,
            "mean_token_accuracy": 0.8,
        }
    )
    assert metrics == {
        "train/loss": 0.5,
        "train/learning_rate": 1e-4,
        "train/epoch": 1.0,
        "train/token_accuracy": 0.8,
    }


def test_prefix_trainer_logs_for_mlflow_maps_val_metrics():
    mlflow_utils = import_training_module("mlflow_utils")
    metrics = mlflow_utils.prefix_trainer_logs_for_mlflow(
        {
            "eval_loss": 0.7,
            "eval_mean_token_accuracy": 0.77,
            "eval_runtime": 12.4,
            "epoch": 2.0,
        }
    )
    assert metrics == {
        "val/loss": 0.7,
        "val/token_accuracy": 0.77,
        "val/runtime": 12.4,
        "val/epoch": 2.0,
    }


def test_prefix_trainer_logs_for_mlflow_maps_train_summary_metrics():
    mlflow_utils = import_training_module("mlflow_utils")
    metrics = mlflow_utils.prefix_trainer_logs_for_mlflow(
        {
            "train_loss": 0.72,
            "train_runtime": 489.3,
            "total_flos": 2.8e16,
        }
    )
    assert metrics == {
        "train/loss": 0.72,
        "train/runtime": 489.3,
        "train/total_flos": 2.8e16,
    }


def test_add_final_summary_val_metrics_includes_last_eval_accuracy():
    mlflow_utils = import_training_module("mlflow_utils")
    logs = {
        "train_loss": 0.72,
        "train_runtime": 489.3,
        "train_samples_per_second": 0.061,
    }
    prefixed = mlflow_utils.prefix_trainer_logs_for_mlflow(logs)
    updated = mlflow_utils.add_final_summary_val_metrics(
        prefixed,
        logs,
        last_val_mean_token_accuracy=0.7053,
    )
    assert updated["val/mean_token_accuracy"] == 0.7053
    assert updated["train/loss"] == 0.72


def test_add_final_summary_val_metrics_skips_non_summary_logs():
    mlflow_utils = import_training_module("mlflow_utils")
    logs = {"loss": 0.5, "learning_rate": 1e-4}
    prefixed = mlflow_utils.prefix_trainer_logs_for_mlflow(logs)
    updated = mlflow_utils.add_final_summary_val_metrics(
        prefixed,
        logs,
        last_val_mean_token_accuracy=0.7053,
    )
    assert updated == prefixed
