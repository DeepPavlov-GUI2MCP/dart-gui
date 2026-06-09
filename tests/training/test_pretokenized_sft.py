from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

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


def test_pretokenized_dataset_record_sharding_matches_rank_length(tmp_path, monkeypatch):
    uitars_collator = import_training_module("uitars_collator")
    pretokenized_dir = tmp_path / "pretokenized"
    pretokenized_dir.mkdir()
    (pretokenized_dir / "manifest.json").write_text('{"expanded_rows": 10}\n', encoding="utf-8")
    records = [
        {
            "task_id": "train-task",
            "features": {
                "input_ids": torch.tensor([idx]),
                "labels": torch.tensor([idx]),
                "attention_mask": torch.tensor([1]),
            },
        }
        for idx in range(10)
    ]
    for shard_idx, record in enumerate(records):
        torch.save([record], pretokenized_dir / f"shard-{shard_idx:05d}.pt")
    (pretokenized_dir / "split.json").write_text(
        """{
  "strategy": "one_holdout_task_per_microaction",
  "holdout_rule": "min_task_index",
  "task_examples_dir": "unused",
  "task_generation_root": null,
  "train_task_ids": ["train-task"],
  "val_task_ids": [],
  "microaction_holdouts": {},
  "train_rows": 8,
  "val_rows": 2
}
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    dataset = uitars_collator.PretokenizedUitarsDataset(pretokenized_dir, split="train")
    assert len(dataset) == 4
    rows = list(DataLoader(dataset, batch_size=None, num_workers=2))
    assert len(rows) == len(dataset)


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
