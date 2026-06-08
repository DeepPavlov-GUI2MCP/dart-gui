from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PRESUBMIT = REPO_ROOT / "jobs" / "sft" / "run_uitars_lora" / "presubmit.py"
FIXTURE_CONFIG = REPO_ROOT / "training" / "configs" / "sft_holo_goal_variants_fixture.yml"
PRETOKENIZED_FIXTURE = REPO_ROOT / "tests" / "training" / "fixtures" / "pretokenized_shards"


def import_training_module(name: str):
    sys.path.insert(0, str(REPO_ROOT / "training"))
    sys.path.insert(0, str(REPO_ROOT))
    try:
        return __import__(name)
    finally:
        if str(REPO_ROOT) in sys.path:
            sys.path.remove(str(REPO_ROOT))
        if str(REPO_ROOT / "training") in sys.path:
            sys.path.remove(str(REPO_ROOT / "training"))


def load_presubmit_module():
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "training"))
    spec = importlib.util.spec_from_file_location("sft_presubmit", PRESUBMIT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load presubmit module from {PRESUBMIT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_fixture_staged_jsonl(path: Path, *, row_count: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for step in range(1, row_count + 1):
        rows.append(
            {
                "task_id": "test-task-001",
                "step_num": step + 3,
                "rollout_dir": str(
                    REPO_ROOT
                    / "tests/training/fixtures/goal_variants/trace_root/rollout/libreoffice_writer/test-task-001"
                ),
                "messages": [],
                "loss_on_last_assistant_only": True,
            }
        )
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_validate_pretokenized_dataset_passes_fixture(tmp_path: Path):
    run_sft = import_training_module("run_sft")
    config = run_sft.load_config_file(str(FIXTURE_CONFIG))
    staged_path = run_sft.resolve_staged_data_path(config)
    write_fixture_staged_jsonl(staged_path, row_count=2)

    presubmit = load_presubmit_module()
    presubmit.validate_pretokenized_dataset(str(FIXTURE_CONFIG))


def test_validate_pretokenized_dataset_fails_row_mismatch(tmp_path: Path):
    run_sft = import_training_module("run_sft")
    config = run_sft.load_config_file(str(FIXTURE_CONFIG))
    staged_path = run_sft.resolve_staged_data_path(config)
    write_fixture_staged_jsonl(staged_path, row_count=3)

    presubmit = load_presubmit_module()
    with pytest.raises(SystemExit) as exc:
        presubmit.validate_pretokenized_dataset(str(FIXTURE_CONFIG))
    assert exc.value.code == 1


def test_presubmit_subprocess_fixture():
    run_sft = import_training_module("run_sft")
    config = run_sft.load_config_file(str(FIXTURE_CONFIG))
    staged_path = run_sft.resolve_staged_data_path(config)
    write_fixture_staged_jsonl(staged_path, row_count=2)

    env = os.environ.copy()
    env["DART_SFT_CONFIG"] = str(FIXTURE_CONFIG)
    result = subprocess.run(
        [sys.executable, str(PRESUBMIT)],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Training rows: 2 (expected 2)" in result.stdout
    assert f"Pretokenized dataset: {PRETOKENIZED_FIXTURE.resolve()}" in result.stdout


def test_collect_pretokenized_stats_fixture():
    manifest_mod = import_training_module("pretokenized_manifest")
    stats = manifest_mod.collect_pretokenized_stats(PRETOKENIZED_FIXTURE)
    assert stats.actual_rows == 2
    assert stats.shard_count == 1
