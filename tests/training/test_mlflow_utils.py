from __future__ import annotations

import os
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


def test_resolve_mlflow_store_uses_shared_backend_and_run_artifacts(tmp_path, monkeypatch):
    mlflow_utils = import_training_module("mlflow_utils")
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("DART_REPO_ROOT", str(repo))
    monkeypatch.delenv("MLFLOW_BACKEND_STORE_URI", raising=False)
    monkeypatch.delenv("MLFLOW_ARTIFACT_ROOT", raising=False)
    run_dir = repo / "outputs" / "sft" / "run-1"
    backend_uri, artifact_root = mlflow_utils.resolve_mlflow_store(output_dir=run_dir)
    assert backend_uri == f"file://{(repo / 'outputs' / 'sft' / 'mlruns').resolve()}"
    assert artifact_root == f"file://{(run_dir / 'mlartifacts').resolve()}"


def test_configure_mlflow_file_tracking_uses_nfs_backend(tmp_path, monkeypatch):
    mlflow_utils = import_training_module("mlflow_utils")
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("DART_REPO_ROOT", str(repo))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    run_dir = repo / "outputs" / "sft" / "run-1"
    tracking_uri = mlflow_utils.configure_mlflow_file_tracking(output_dir=run_dir)
    assert tracking_uri == f"file://{(repo / 'outputs' / 'sft' / 'mlruns').resolve()}"
    assert os.environ["MLFLOW_TRACKING_URI"] == tracking_uri
    assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "true"


def test_configure_mlflow_uses_file_tracking_by_default(tmp_path, monkeypatch):
    mlflow_utils = import_training_module("mlflow_utils")
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("DART_REPO_ROOT", str(repo))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    settings = mlflow_utils.MlflowSettings(enabled=True, experiment_name="exp-a")
    assert mlflow_utils.configure_mlflow(settings, output_dir=repo / "outputs" / "run-1")
    assert os.environ["MLFLOW_TRACKING_URI"].startswith("file://")
    assert os.environ["MLFLOW_EXPERIMENT_NAME"] == "exp-a"


def test_prepare_mlflow_file_store_creates_directories(tmp_path, monkeypatch):
    mlflow_utils = import_training_module("mlflow_utils")
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("DART_REPO_ROOT", str(repo))
    run_dir = repo / "outputs" / "sft" / "run-1"
    backend_uri, artifact_root = mlflow_utils.prepare_mlflow_file_store(output_dir=run_dir)
    assert Path(backend_uri.removeprefix("file://")).is_dir()
    assert Path(artifact_root.removeprefix("file://")).is_dir()
    assert os.environ["MLFLOW_BACKEND_STORE_URI"] == backend_uri


def test_resolve_mlflow_tmux_session_default():
    mlflow_utils = import_training_module("mlflow_utils")
    assert mlflow_utils.resolve_mlflow_tmux_session() == "dart-sft-mlflow"


def test_build_mlflow_server_shell_command_includes_file_backend():
    mlflow_utils = import_training_module("mlflow_utils")
    cmd = mlflow_utils._build_mlflow_server_shell_command(
        backend_uri="file:///tmp/mlruns",
        artifact_root="file:///tmp/mlartifacts",
        host="127.0.0.1",
        port=5000,
        python=Path("/tmp/.venv/bin/python"),
        repo_root=Path("/tmp/repo"),
    )
    assert "MLFLOW_ALLOW_FILE_STORE=true" in cmd
    assert "file:///tmp/mlruns" in cmd
    assert "/tmp/mlartifacts" in cmd


def test_resolve_mlflow_connect_host_uses_localhost_for_single_node_jobs(monkeypatch):
    mlflow_utils = import_training_module("mlflow_utils")
    monkeypatch.delenv("MLFLOW_HOST", raising=False)
    monkeypatch.setenv("DART_JOB_NUM_NODES", "1")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("MASTER_ADDR", "mpimaster-0.example")
    assert mlflow_utils.resolve_mlflow_connect_host() == "127.0.0.1"


def test_resolve_mlflow_connect_host_uses_master_addr_for_workers(monkeypatch):
    mlflow_utils = import_training_module("mlflow_utils")
    monkeypatch.delenv("MLFLOW_HOST", raising=False)
    monkeypatch.setenv("DART_JOB_NUM_NODES", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("MASTER_ADDR", "mpimaster-0.example")
    assert mlflow_utils.resolve_mlflow_connect_host() == "mpimaster-0.example"


def test_is_mlflow_server_reachable_false_for_closed_port():
    mlflow_utils = import_training_module("mlflow_utils")
    assert mlflow_utils.is_mlflow_server_reachable("http://127.0.0.1:1") is False


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
