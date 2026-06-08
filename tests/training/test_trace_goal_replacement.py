from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "training"))

from trace_goal_replacement import (  # noqa: E402
    GoalReplaceSettings,
    build_augmented_task,
    load_goal_variants,
    load_task_payload,
    run_goal_replacement,
)

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "trace_goal_replacement"
TRACE_ROOT = FIXTURE_ROOT / "trace_root" / "rollout"
GOALS_ROOT = FIXTURE_ROOT / "goals"
GENERATION_RUN = FIXTURE_ROOT


def _task_examples_dir(tmp_path: Path) -> Path:
    source = FIXTURE_ROOT / "task_examples"
    dest = tmp_path / "task_examples"
    dest.mkdir(parents=True)
    for path in source.rglob("*.json"):
        rel = path.relative_to(source)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        text = path.read_text(encoding="utf-8").replace(
            "__GENERATION_RUN__",
            str(GENERATION_RUN),
        )
        target.write_text(text, encoding="utf-8")
    return dest


@pytest.fixture
def settings(tmp_path: Path) -> GoalReplaceSettings:
    return GoalReplaceSettings(
        trace_root=TRACE_ROOT,
        task_examples_dir=_task_examples_dir(tmp_path),
        output_dir=tmp_path / "output",
        max_variants=1,
        goals_root=GOALS_ROOT.parent,
        dry_run=False,
        overwrite=True,
    )


def test_build_augmented_task_replaces_instruction() -> None:
    base_task = {
        "id": "test-task-001",
        "instruction": "Open Formula from the New submenu.",
        "synthetic": {"task_index": 0},
    }
    variants = load_goal_variants(GOALS_ROOT / "test-task-001", "test-task-001", max_variants=1)
    augmented = build_augmented_task(base_task, variants[0])
    assert augmented["instruction"] == "Open a blank formula document"
    assert augmented["synthetic"]["goal_variant_index"] == 0
    assert augmented["synthetic"]["original_instruction"] == "Open Formula from the New submenu."
    assert augmented["synthetic"]["goal_variant_expected_outcome"] == "A formula document opens."


def test_default_max_variants_writes_only_variant_0(settings: GoalReplaceSettings) -> None:
    summary = run_goal_replacement(settings)
    assert summary["processed_ok"] == 1
    assert summary["variants_written"] == 1
    output = settings.output_dir / "variant_0" / "libreoffice_writer" / "test-task-001.json"
    assert output.is_file()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["instruction"] == "Open a blank formula document"
    for idx in (1, 2, 3):
        assert not (settings.output_dir / f"variant_{idx}" / "libreoffice_writer" / "test-task-001.json").exists()


def test_max_variants_four_writes_all_variants(settings: GoalReplaceSettings) -> None:
    settings = GoalReplaceSettings(
        trace_root=settings.trace_root,
        task_examples_dir=settings.task_examples_dir,
        output_dir=settings.output_dir,
        max_variants=4,
        goals_root=settings.goals_root,
        task_ids=("test-task-001",),
        dry_run=False,
        overwrite=True,
    )
    summary = run_goal_replacement(settings)
    assert summary["processed_ok"] == 1
    assert summary["variants_written"] == 4
    for idx, expected in enumerate(
        [
            "Open a blank formula document",
            "Make a new math formula",
            "Start a LibreOffice Math file",
            "Create an empty formula document",
        ]
    ):
        output = settings.output_dir / f"variant_{idx}" / "libreoffice_writer" / "test-task-001.json"
        assert output.is_file()
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["instruction"] == expected


def test_id_mismatch_is_reported_as_failure(settings: GoalReplaceSettings) -> None:
    summary = run_goal_replacement(
        GoalReplaceSettings(
            trace_root=settings.trace_root,
            task_examples_dir=settings.task_examples_dir,
            output_dir=settings.output_dir,
            max_variants=1,
            goals_root=settings.goals_root,
            task_ids=("test-task-mismatch",),
            dry_run=False,
            overwrite=True,
        )
    )
    assert summary["processed_ok"] == 0
    assert len(summary["failures"]) == 1
    assert "mismatch" in summary["failures"][0]["error"].lower()


def test_missing_goals_are_skipped(settings: GoalReplaceSettings) -> None:
    summary = run_goal_replacement(
        GoalReplaceSettings(
            trace_root=settings.trace_root,
            task_examples_dir=settings.task_examples_dir,
            output_dir=settings.output_dir,
            max_variants=1,
            goals_root=settings.goals_root,
            task_ids=("test-task-no-goals",),
            dry_run=False,
            overwrite=True,
        )
    )
    assert summary["processed_ok"] == 0
    assert summary["skipped_no_goals"] == 1


def test_require_all_goals_fails_when_variants_missing(settings: GoalReplaceSettings) -> None:
    summary = run_goal_replacement(
        GoalReplaceSettings(
            trace_root=settings.trace_root,
            task_examples_dir=settings.task_examples_dir,
            output_dir=settings.output_dir,
            max_variants=4,
            goals_root=settings.goals_root,
            task_ids=("test-task-no-goals",),
            dry_run=False,
            overwrite=True,
            require_all_goals=True,
        )
    )
    assert summary["require_all_goals_failed"] is True
    assert summary["failures"]


def test_load_task_payload_validates_id(tmp_path: Path) -> None:
    task_examples = _task_examples_dir(tmp_path)
    with pytest.raises(ValueError, match="mismatch"):
        load_task_payload(task_examples, "libreoffice_writer", "test-task-mismatch")
