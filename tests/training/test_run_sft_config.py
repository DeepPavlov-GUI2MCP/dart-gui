from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SFT = REPO_ROOT / "training" / "run_sft.py"
SFT_CONFIG = REPO_ROOT / "training" / "configs" / "sft_example.yml"


def test_resolve_config_from_example():
    sys.path.insert(0, str(REPO_ROOT / "training"))
    try:
        import run_sft
    finally:
        sys.path.pop(0)

    config = run_sft.load_config_file(str(SFT_CONFIG))
    assert config.model.base_model == "ByteDance-Seed/UI-TARS-1.5-7B"
    assert config.model.max_seq_length == 2048
    assert config.model.load_in_4bit is True
    assert config.dataset.repo_id == "mlabonne/FineTome-100k"
    assert config.dataset.field_messages == "conversations"
    assert config.lora.r == 16
    assert config.training.max_steps == 5
    assert config.output_dir == (REPO_ROOT / "outputs/sft/ui-tars-example").resolve()


def test_normalize_messages_with_mappings():
    sys.path.insert(0, str(REPO_ROOT / "training"))
    try:
        import run_sft
    finally:
        sys.path.pop(0)

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


def test_dry_run_subprocess():
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
    assert "Staged dataset:" in result.stdout
    assert "ByteDance-Seed/UI-TARS-1.5-7B" in result.stdout
