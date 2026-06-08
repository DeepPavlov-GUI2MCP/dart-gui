from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests/training/fixtures/goal_variants"


def import_training_module(name: str):
    sys.path.insert(0, str(REPO_ROOT / "training"))
    try:
        return __import__(name)
    finally:
        sys.path.pop(0)


def test_find_latest_generation_run_picks_newest_timestamp():
    goal_variants = import_training_module("goal_variants")
    root = FIXTURE_ROOT / "task_generation"
    latest = goal_variants.find_latest_generation_run(root, "macro-test-001")
    assert latest.name == "20260201T120000Z"


def test_load_variants_for_task_uses_latest_goals():
    goal_variants = import_training_module("goal_variants")
    root = FIXTURE_ROOT / "task_generation"
    task_examples = FIXTURE_ROOT / "task_examples"
    variants = goal_variants.load_variants_for_task(
        root,
        task_examples,
        "libreoffice_writer",
        "test-task-001",
        max_variants=2,
    )
    assert len(variants) == 2
    assert variants[0].goal == "Open a blank formula document"
    assert variants[1].goal == "Create a new formula file"


def test_replace_instruction_in_messages_swaps_first_user_text():
    goal_variants = import_training_module("goal_variants")
    uitars_format = import_training_module("uitars_format")
    settings = uitars_format.UitarsFormatSettings()
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "system"}]},
        {"role": "user", "content": [{"type": "text", "text": "old prompt"}]},
        {"role": "user", "content": [{"type": "image", "path": "step.png"}]},
    ]
    updated = goal_variants.replace_instruction_in_messages(
        messages,
        "New user goal",
        settings,
    )
    assert updated[1]["content"][0]["text"] != messages[1]["content"][0]["text"]
    assert "New user goal" in updated[1]["content"][0]["text"]
    assert updated[2] == messages[2]


def test_assert_tasks_have_goal_variants_raises_when_missing():
    goal_variants = import_training_module("goal_variants")
    root = FIXTURE_ROOT / "task_generation"
    task_examples = FIXTURE_ROOT / "task_examples"
    rows = [
        {
            "task_id": "test-task-missing",
            "rollout_dir": str(
                FIXTURE_ROOT / "trace_root/rollout/libreoffice_writer/test-task-missing"
            ),
        }
    ]
    with pytest.raises(ValueError, match="missing goal variants"):
        goal_variants.assert_tasks_have_goal_variants(
            root,
            task_examples,
            rows,
            max_variants=1,
        )


def test_assert_tasks_have_goal_variants_passes_for_cataloged_task():
    goal_variants = import_training_module("goal_variants")
    root = FIXTURE_ROOT / "task_generation"
    task_examples = FIXTURE_ROOT / "task_examples"
    rows = [
        {
            "task_id": "test-task-001",
            "rollout_dir": str(
                FIXTURE_ROOT / "trace_root/rollout/libreoffice_writer/test-task-001"
            ),
        }
    ]
    catalog = goal_variants.assert_tasks_have_goal_variants(
        root,
        task_examples,
        rows,
        max_variants=1,
    )
    assert "test-task-001" in catalog
    assert catalog["test-task-001"][0].goal == "Open a blank formula document"
