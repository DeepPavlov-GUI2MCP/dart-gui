from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SFT = REPO_ROOT / "training" / "run_sft.py"
PRETOKENIZED_FIXTURE = REPO_ROOT / "tests/training/fixtures/pretokenized_shards"
SFT_GOAL_CONFIG = REPO_ROOT / "training/configs/sft_holo_goal_variants_fixture.yml"


def import_training_module(name: str):
    sys.path.insert(0, str(REPO_ROOT / "training"))
    try:
        return __import__(name)
    finally:
        sys.path.pop(0)


def test_pretokenized_dataset_loads_shard_records():
    uitars_collator = import_training_module("uitars_collator")
    dataset = uitars_collator.PretokenizedUitarsDataset(PRETOKENIZED_FIXTURE)
    assert len(dataset) == 2
    item = dataset[0]
    assert "features" in item
    assert item["features"]["input_ids"].shape[-1] == 5


def test_pretokenized_collator_pads_batch():
    uitars_collator = import_training_module("uitars_collator")
    dataset = uitars_collator.PretokenizedUitarsDataset(PRETOKENIZED_FIXTURE)
    collator = uitars_collator.make_pretokenized_collator(pad_token_id=0)
    batch = collator([dataset[0], dataset[1]])
    assert batch["input_ids"].shape == (2, 7)
    assert batch["labels"].shape == (2, 7)
    assert batch["attention_mask"].shape == (2, 7)


def test_dry_run_from_pretokenized_traces_subprocess():
    result = subprocess.run(
        [
            sys.executable,
            str(RUN_SFT),
            "--config",
            str(SFT_GOAL_CONFIG),
            "--from-pretokenized-traces",
            str(PRETOKENIZED_FIXTURE),
            "--dry-run",
            "--skip-hf-validation",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Pretokenized dataset:" in result.stdout
    assert str(PRETOKENIZED_FIXTURE) in result.stdout
