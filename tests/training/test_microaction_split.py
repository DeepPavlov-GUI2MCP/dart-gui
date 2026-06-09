from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_EXAMPLES = REPO_ROOT / "GUI-Docker-Env/evaluation_examples/examples"


def import_training_module(name: str):
    sys.path.insert(0, str(REPO_ROOT / "training"))
    try:
        return __import__(name)
    finally:
        sys.path.pop(0)


def test_compute_microaction_split_holds_out_min_task_index():
    microaction_split = import_training_module("microaction_split")
    rows = [
        {"task_id": "e2f6d988-9db1-5970-b79c-c1aff9f387f0", "rollout_dir": "libreoffice_writer/e2f6d988-9db1-5970-b79c-c1aff9f387f0"},
        {"task_id": "9825624b-ea68-52db-916f-ee56760b914d", "rollout_dir": "libreoffice_writer/9825624b-ea68-52db-916f-ee56760b914d"},
    ]
    split = microaction_split.compute_microaction_split(
        rows,
        task_examples_dir=str(TASK_EXAMPLES),
        task_generation_root="../gui_distillation/data/synthetic/libreoffice_writer/active_root_231/task_generation/gpt-5.4",
        rows_per_task={"e2f6d988-9db1-5970-b79c-c1aff9f387f0": 4, "9825624b-ea68-52db-916f-ee56760b914d": 3},
    )
    assert "9825624b-ea68-52db-916f-ee56760b914d" in split.val_task_ids
    assert "e2f6d988-9db1-5970-b79c-c1aff9f387f0" in split.train_task_ids
    assert split.val_rows == 3
    assert split.train_rows == 4


def test_compute_smoke_microaction_split_uses_four_tasks_per_microaction():
    microaction_split = import_training_module("microaction_split")
    rows = []
    rows_per_task = {}
    for micro_idx in range(5):
        for task_idx in range(5):
            task_id = f"task-{micro_idx}-{task_idx}"
            rows.append({"task_id": task_id, "rollout_dir": f"libreoffice_writer/{task_id}"})
            rows_per_task[task_id] = task_idx + 1
    original = microaction_split.load_task_microaction_meta

    def fake_meta(*, task_examples_dir, domain, task_id, task_generation_root=None):
        _, micro_idx, task_idx = task_id.split("-", 2)
        return microaction_split.TaskMicroactionMeta(
            task_id=task_id,
            domain=domain,
            micro_action_id=f"micro-{micro_idx}",
            task_index=int(task_idx),
        )

    microaction_split.load_task_microaction_meta = fake_meta
    try:
        split = microaction_split.compute_smoke_microaction_split(
            rows,
            task_examples_dir=str(TASK_EXAMPLES),
            task_generation_root=None,
            rows_per_task=rows_per_task,
            num_microactions=4,
        )
    finally:
        microaction_split.load_task_microaction_meta = original
    assert split.strategy == microaction_split.SMOKE_SPLIT_STRATEGY
    assert len(split.val_task_ids) == 4
    assert len(split.train_task_ids) == 12
    assert len(set(split.train_task_ids).intersection(split.val_task_ids)) == 0
    assert split.train_rows == sum(rows_per_task[task_id] for task_id in split.train_task_ids)
    assert split.val_rows == sum(rows_per_task[task_id] for task_id in split.val_task_ids)


def test_validate_split_for_training_rows_accepts_matching_split():
    microaction_split = import_training_module("microaction_split")
    rows = [
        {"task_id": "e2f6d988-9db1-5970-b79c-c1aff9f387f0", "rollout_dir": "libreoffice_writer/e2f6d988-9db1-5970-b79c-c1aff9f387f0"},
        {"task_id": "9825624b-ea68-52db-916f-ee56760b914d", "rollout_dir": "libreoffice_writer/9825624b-ea68-52db-916f-ee56760b914d"},
    ]
    kwargs = {
        "task_examples_dir": str(TASK_EXAMPLES),
        "task_generation_root": "../gui_distillation/data/synthetic/libreoffice_writer/active_root_231/task_generation/gpt-5.4",
    }
    split = microaction_split.compute_split_for_training_rows(rows, **kwargs)
    microaction_split.validate_split_for_training_rows(split, rows, **kwargs)


def test_validate_split_for_training_rows_rejects_mismatched_holdout():
    microaction_split = import_training_module("microaction_split")
    rows = [
        {"task_id": "e2f6d988-9db1-5970-b79c-c1aff9f387f0", "rollout_dir": "libreoffice_writer/e2f6d988-9db1-5970-b79c-c1aff9f387f0"},
        {"task_id": "9825624b-ea68-52db-916f-ee56760b914d", "rollout_dir": "libreoffice_writer/9825624b-ea68-52db-916f-ee56760b914d"},
    ]
    split = microaction_split.compute_microaction_split(
        rows,
        task_examples_dir=str(TASK_EXAMPLES),
        task_generation_root="../gui_distillation/data/synthetic/libreoffice_writer/active_root_231/task_generation/gpt-5.4",
        rows_per_task={"e2f6d988-9db1-5970-b79c-c1aff9f387f0": 4, "9825624b-ea68-52db-916f-ee56760b914d": 3},
    )
    bad = microaction_split.MicroactionSplit(
        strategy=split.strategy,
        holdout_rule=split.holdout_rule,
        task_examples_dir=split.task_examples_dir,
        task_generation_root=split.task_generation_root,
        train_task_ids=split.val_task_ids,
        val_task_ids=split.train_task_ids,
        microaction_holdouts=split.microaction_holdouts,
        train_rows=split.val_rows,
        val_rows=split.train_rows,
    )
    with pytest.raises(ValueError, match="microaction holdout rule"):
        microaction_split.validate_split_for_training_rows(
            bad,
            rows,
            task_examples_dir=str(TASK_EXAMPLES),
            task_generation_root="../gui_distillation/data/synthetic/libreoffice_writer/active_root_231/task_generation/gpt-5.4",
        )


def test_rows_per_task_from_catalog_counts_steps_times_variants():
    microaction_split = import_training_module("microaction_split")
    rows = [
        {"task_id": "task-a"},
        {"task_id": "task-a"},
        {"task_id": "task-b"},
    ]
    counts = microaction_split.rows_per_task_from_catalog(
        rows,
        goal_variants=True,
        catalog={"task-a": [1, 2], "task-b": [1]},
    )
    assert counts == {"task-a": 4, "task-b": 1}
