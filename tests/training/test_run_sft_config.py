from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SFT = REPO_ROOT / "training" / "run_sft.py"
SFT_CONFIG = REPO_ROOT / "training" / "configs" / "sft_example.yml"
SFT_CHAT_CONFIG = REPO_ROOT / "training" / "configs" / "sft_example_chat.yml"
SFT_FIXTURE_CONFIG = REPO_ROOT / "training" / "configs" / "sft_fixture.yml"
FIXTURE_ROLLOUT = (
    REPO_ROOT
    / "tests/training/fixtures/uitars_trace/rollout/libreoffice_writer/test-task-id"
)


def import_training_module(name: str):
    sys.path.insert(0, str(REPO_ROOT / "training"))
    try:
        return __import__(name)
    finally:
        sys.path.pop(0)


def test_resolve_config_from_example():
    run_sft = import_training_module("run_sft")
    config = run_sft.load_config_file(str(SFT_CONFIG))
    assert config.model.base_model == "ByteDance-Seed/UI-TARS-1.5-7B"
    assert config.model.max_seq_length == 2048
    assert config.training.fp16 is True
    assert config.training.bf16 is False
    assert config.dataset.format == "uitars_trace"
    assert config.dataset.sample_mode == "per_step"
    assert config.dataset.history_n == 5
    assert config.dataset.min_result == 1.0
    assert config.lora.r == 16
    assert config.training.max_steps == 5
    assert config.training.packing is False
    assert config.output_dir == (REPO_ROOT / "outputs/sft/ui-tars-example").resolve()


def test_resolve_chat_config_from_example():
    run_sft = import_training_module("run_sft")
    config = run_sft.load_config_file(str(SFT_CHAT_CONFIG))
    assert config.dataset.format == "chat"
    assert config.dataset.repo_id == "mlabonne/FineTome-100k"
    assert config.dataset.field_messages == "conversations"


def test_normalize_messages_with_mappings():
    run_sft = import_training_module("run_sft")
    dataset = run_sft.DatasetSettings(
        repo_id="dummy",
        field_messages="conversations",
        message_property_mappings={"role": "from", "content": "value"},
    )
    row = {
        "conversations": [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi"},
        ]
    }
    messages = run_sft.normalize_messages(row, dataset)
    assert messages == [
        {"role": "human", "content": "Hello"},
        {"role": "gpt", "content": "Hi"},
    ]


def test_build_per_step_rows_from_fixture():
    uitars_trace_dataset = import_training_module("uitars_trace_dataset")
    uitars_format = import_training_module("uitars_format")
    settings = uitars_trace_dataset.TraceScanSettings(
        trace_roots=(str(REPO_ROOT / "tests/training/fixtures/uitars_trace/rollout"),),
        task_examples_dir=str(REPO_ROOT / "tests/training/fixtures/uitars_trace/task_examples"),
        sample_mode="per_step",
        history_n=5,
        uitars=uitars_format.UitarsFormatSettings(),
    )
    rows = uitars_trace_dataset.build_rows_for_rollout(FIXTURE_ROLLOUT, settings)
    assert len(rows) == 2
    for row in rows:
        messages = row["messages"]
        roles = [message["role"] for message in messages]
        assert roles[0] == "system"
        assert roles[1] == "user"
        assert roles.count("assistant") >= 1
        assert roles[-1] == "assistant"
        assert row["loss_on_last_assistant_only"] is True


def test_collator_masks_labels_outside_last_assistant():
    import torch

    uitars_collator = import_training_module("uitars_collator")
    processor = MagicMock()
    processor.apply_chat_template.side_effect = [
        {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7]])},
        {"input_ids": torch.tensor([[1, 2, 3, 4]])},
    ]
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "old"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "target"}]},
    ]
    batch = uitars_collator.tokenize_uitars_messages(
        processor,
        messages,
        loss_on_last_assistant_only=True,
    )
    labels = batch["labels"].tolist()
    assert labels[:4] == [-100, -100, -100, -100]
    assert labels[4:] == [5, 6, 7]


def test_dry_run_subprocess_fixture():
    result = subprocess.run(
        [
            sys.executable,
            str(RUN_SFT),
            "--config",
            str(SFT_FIXTURE_CONFIG),
            "--dry-run",
            "--skip-hf-validation",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Staged dataset:" in result.stdout
    assert "uitars_trace" in result.stdout


def test_dry_run_subprocess_example_config():
    result = subprocess.run(
        [
            sys.executable,
            str(RUN_SFT),
            "--config",
            str(SFT_CONFIG),
            "--dry-run",
            "--skip-hf-validation",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Dataset format: uitars_trace" in result.stdout
