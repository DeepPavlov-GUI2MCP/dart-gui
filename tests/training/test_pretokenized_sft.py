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


def _write_split_pretokenized_dir(
    pretokenized_dir: Path,
    *,
    train_records: list[dict],
    val_records: list[dict],
) -> None:
    pretokenized_dir.mkdir(parents=True, exist_ok=True)
    all_records = train_records + val_records
    (pretokenized_dir / "manifest.json").write_text(
        f'{{"expanded_rows": {len(all_records)}}}\n',
        encoding="utf-8",
    )
    for shard_idx, record in enumerate(all_records):
        torch.save([record], pretokenized_dir / f"shard-{shard_idx:05d}.pt")
    (pretokenized_dir / "split.json").write_text(
        f"""{{
  "strategy": "one_holdout_task_per_microaction",
  "holdout_rule": "min_task_index",
  "task_examples_dir": "unused",
  "task_generation_root": null,
  "train_task_ids": ["train-task"],
  "val_task_ids": ["val-task"],
  "microaction_holdouts": {{}},
  "train_rows": {len(train_records)},
  "val_rows": {len(val_records)}
}}
""",
        encoding="utf-8",
    )


def _record(idx: int, task_id: str) -> dict:
    return {
        "task_id": task_id,
        "features": {
            "input_ids": torch.tensor([idx]),
            "labels": torch.tensor([idx]),
            "attention_mask": torch.tensor([1]),
        },
    }


def _collect_rank_items(dataset, *, num_workers: int = 0) -> list[dict]:
    rows = list(DataLoader(dataset, batch_size=None, num_workers=num_workers))
    return sorted(rows, key=lambda item: int(item["features"]["input_ids"].item()))


def test_pretokenized_shard_index_matches_modulo_semantics(tmp_path, monkeypatch):
    uitars_collator = import_training_module("uitars_collator")
    pretokenized_shard_index = import_training_module("pretokenized_shard_index")
    pretokenized_dir = tmp_path / "pretokenized"
    train_records = [_record(idx, "train-task") for idx in range(8)]
    val_records = [_record(100 + idx, "val-task") for idx in range(4)]
    _write_split_pretokenized_dir(
        pretokenized_dir,
        train_records=train_records,
        val_records=val_records,
    )
    world_size = 4
    pretokenized_shard_index.ensure_shard_index(
        pretokenized_dir,
        split_name="train",
        allowed_task_ids={"train-task"},
        world_size=world_size,
    )
    for rank in range(world_size):
        monkeypatch.setenv("RANK", str(rank))
        monkeypatch.setenv("WORLD_SIZE", str(world_size))
        indexed = _collect_rank_items(
            uitars_collator.PretokenizedUitarsDataset(pretokenized_dir, split="train")
        )
        monkeypatch.setattr(uitars_collator, "load_shard_index", lambda *_a, **_k: None)
        legacy = _collect_rank_items(
            uitars_collator.PretokenizedUitarsDataset(pretokenized_dir, split="train")
        )
        assert indexed == legacy


def test_pretokenized_shard_index_opens_only_assigned_shards(tmp_path, monkeypatch):
    uitars_collator = import_training_module("uitars_collator")
    pretokenized_shard_index = import_training_module("pretokenized_shard_index")
    pretokenized_dir = tmp_path / "pretokenized"
    train_records = [_record(idx, "train-task") for idx in range(10)]
    val_records = [_record(100 + idx, "val-task") for idx in range(2)]
    _write_split_pretokenized_dir(
        pretokenized_dir,
        train_records=train_records,
        val_records=val_records,
    )
    world_size = 2
    pretokenized_shard_index.ensure_shard_index(
        pretokenized_dir,
        split_name="train",
        allowed_task_ids={"train-task"},
        world_size=world_size,
    )
    opened: list[str] = []
    original_loader = uitars_collator._load_shard_records

    def tracking_loader(shard_path: Path):
        opened.append(shard_path.name)
        return original_loader(shard_path)

    monkeypatch.setattr(uitars_collator, "_load_shard_records", tracking_loader)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    list(DataLoader(uitars_collator.PretokenizedUitarsDataset(pretokenized_dir, split="train")))
    assert opened
    assert len(opened) < len(train_records) + len(val_records)
    assert all(name in {f"shard-{idx:05d}.pt" for idx in (0, 2, 4, 6, 8)} for name in opened)


def test_pretokenized_shard_index_fallback_without_index(tmp_path, monkeypatch):
    uitars_collator = import_training_module("uitars_collator")
    pretokenized_dir = tmp_path / "pretokenized"
    train_records = [_record(idx, "train-task") for idx in range(6)]
    _write_split_pretokenized_dir(
        pretokenized_dir,
        train_records=train_records,
        val_records=[],
    )
    (pretokenized_dir / "split.json").write_text(
        (pretokenized_dir / "split.json").read_text(encoding="utf-8").replace(
            '"val_task_ids": ["val-task"]',
            '"val_task_ids": []',
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    dataset = uitars_collator.PretokenizedUitarsDataset(pretokenized_dir, split="train")
    assert len(dataset) == 3
    rows = _collect_rank_items(dataset)
    assert len(rows) == 3
    assert [int(item["features"]["input_ids"].item()) for item in rows] == [1, 3, 5]


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
