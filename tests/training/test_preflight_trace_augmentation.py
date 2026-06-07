from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "training"))
sys.path.insert(0, str(REPO_ROOT / "GUI-Docker-Env"))

from preflight_trace_augmentation import (  # noqa: E402
    COST_AUGMENTATION_LOG,
    AugmentSettings,
    augmented_task_path,
    bbox_center_to_holo_coords,
    fallback_tool_call,
    generate_holo_native_response,
    hash_preflight_steps,
    load_task_metadata,
    normalize_oracle_step_response,
    normalize_step_response,
    observed_preflight_rows,
    run_augmentation,
    select_manifest_entries,
    discover_manifest_entries,
)
from uitars_trace_dataset import load_traj_rows  # noqa: E402

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "holo_preflight"
TRACE_ROOT = FIXTURE_ROOT / "trace_root"
TASK_EXAMPLES = FIXTURE_ROOT / "task_examples"


@pytest.fixture
def oracle_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PREFLIGHT_ORACLE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("PREFLIGHT_ORACLE_OPENAI_MODEL", "test-model")
    monkeypatch.setenv("PREFLIGHT_ORACLE_OPENAI_BASE_URL", "https://example.com/v1")


def test_cache_key_stable_for_shared_macrostate() -> None:
    meta_a = load_task_metadata(TASK_EXAMPLES, "libreoffice_writer", "task-a")
    meta_b = load_task_metadata(TASK_EXAMPLES, "libreoffice_writer", "task-b")
    assert meta_a.cache_key == meta_b.cache_key
    assert meta_a.macro_state_id == "test-macro-001"


def test_fallback_tool_call_uses_bbox_center() -> None:
    step = {
        "op": "click",
        "meta": {"recorded_bbox": [10, 20, 40, 20], "role": "menu", "name": "Table"},
    }
    screenshot = TRACE_ROOT / "rollout/libreoffice_writer/task-a/step_3_20250101@120002.png"
    tool_call = fallback_tool_call(step, screenshot)
    x, y = bbox_center_to_holo_coords([10, 20, 40, 20], 100, 80)
    assert tool_call["tool_name"] == "click"
    assert tool_call["x"] == x
    assert tool_call["y"] == y


def test_normalize_step_response_fills_invalid_tool_call() -> None:
    step = {
        "op": "wait_for",
        "selector": {"role": "frame", "name_contains": "LibreOffice Writer"},
    }
    screenshot = TRACE_ROOT / "rollout/libreoffice_writer/task-a/step_2_20250101@120001.png"
    response = normalize_step_response(
        {"thought": "Wait for Writer.", "tool_call": {"tool_name": "click", "x": 9999, "y": -1}},
        config_step=step,
        screenshot_path=screenshot,
    )
    payload = json.loads(response)
    assert payload["thought"] == "Wait for Writer."
    assert payload["tool_call"]["tool_name"] == "wait"


def test_normalize_oracle_step_response_ignores_llm_tool_call() -> None:
    step = {
        "op": "click",
        "meta": {"recorded_bbox": [10, 20, 40, 20], "role": "menu", "name": "Table"},
    }
    screenshot = TRACE_ROOT / "rollout/libreoffice_writer/task-a/step_3_20250101@120002.png"
    response = normalize_oracle_step_response(
        {
            "thought": "Open the Table menu.",
            "tool_call": {"tool_name": "click", "x": 300, "y": 375, "button": "left"},
        },
        config_step=step,
        screenshot_path=screenshot,
        step_index=1,
    )
    payload = json.loads(response)
    expected = json.loads(
        normalize_oracle_step_response(
            {"thought": "Open the Table menu."},
            config_step=step,
            screenshot_path=screenshot,
            step_index=1,
        )
    )
    assert payload["thought"] == "Open the Table menu."
    assert payload["tool_call"] == expected["tool_call"]
    x, y = bbox_center_to_holo_coords([10, 20, 40, 20], 100, 80)
    assert payload["tool_call"]["x"] == x
    assert payload["tool_call"]["y"] == y


def test_oracle_cache_dedup_across_tasks(tmp_path: Path, oracle_env: None) -> None:
    trace_root = tmp_path / "trace"
    trace_root.mkdir()
    for task_id in ("task-a", "task-b"):
        src = TRACE_ROOT / "rollout/libreoffice_writer" / task_id
        dst = trace_root / "rollout/libreoffice_writer" / task_id
        dst.mkdir(parents=True, exist_ok=True)
        for path in src.iterdir():
            if path.is_file():
                dst.joinpath(path.name).write_bytes(path.read_bytes())

    oracle_payload = {
        "steps": [
            {
                "preflight_step_index": 0,
                "thought": "Wait for Writer to appear.",
            },
            {
                "preflight_step_index": 1,
                "thought": "Open the Table menu.",
            },
        ]
    }

    with patch("preflight_trace_augmentation.post_chat_completion") as mock_post:
        mock_post.return_value = {
            "choices": [{"message": {"content": json.dumps(oracle_payload)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cost": 0.001},
        }
        settings = AugmentSettings(
            trace_root=trace_root,
            task_examples_dir=TASK_EXAMPLES,
            mode="openai-api",
            n_tasks=2,
            overwrite=True,
        )
        summary = run_augmentation(settings)

    assert summary["ok"] is True
    assert summary["llm_calls"] == 1
    assert summary["cache_hits"] == 1
    assert mock_post.call_count == 1

    task_a = json.loads(
        (trace_root / "preflight_augmented/openai_api/test-model/task-a.json").read_text(encoding="utf-8")
    )
    task_b = json.loads(
        (trace_root / "preflight_augmented/openai_api/test-model/task-b.json").read_text(encoding="utf-8")
    )
    assert len(task_a["steps"]) == 2
    assert len(task_b["steps"]) == 2
    assert task_a["source"] in {"llm", "cache+llm"}
    assert task_b["source"] == "cache"
    step_one = json.loads(task_a["steps"][1]["response"])
    x, y = bbox_center_to_holo_coords([10, 20, 40, 20], 100, 80)
    assert step_one["thought"] == "Open the Table menu."
    assert step_one["tool_call"]["x"] == x
    assert step_one["tool_call"]["y"] == y

    cost_log = trace_root / COST_AUGMENTATION_LOG
    assert cost_log.is_file()
    cost_lines = [json.loads(line) for line in cost_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(cost_lines) == 1
    assert cost_lines[0]["task_id"] == "task-a"
    assert cost_lines[0]["cost"] == 0.001
    assert cost_lines[0]["num_tokens"] == 150
    assert summary["total_cost"] == 0.001


def test_holo_native_prompt_uses_setup_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOLO_OPENAI_MODEL", "test-holo")
    rollout_dir = TRACE_ROOT / "rollout/libreoffice_writer/task-a"
    metadata = load_task_metadata(TASK_EXAMPLES, "libreoffice_writer", "task-a")
    observed = observed_preflight_rows(load_traj_rows(rollout_dir))
    response_payload = {
        "thought": "Wait for LibreOffice Writer to finish opening.",
        "tool_call": {"tool_name": "wait"},
    }

    with patch("preflight_trace_augmentation.post_chat_completion") as mock_post:
        mock_post.return_value = {
            "choices": [{"message": {"content": json.dumps(response_payload)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        response, cost_usd = generate_holo_native_response(
            metadata,
            rollout_dir,
            observed[0],
            completed_indices=[],
            config={
                "api_key": "empty",
                "model": "test-holo",
                "base_url": "http://127.0.0.1:8010/v1",
                "timeout": 1,
                "max_retries": 1,
                "max_pixels": 2116800,
                "min_pixels": 3136,
            },
        )

    body = mock_post.call_args.args[1]
    observation_content = body["messages"][1]["content"]
    image_chunks = [item for item in observation_content if item.get("type") == "image_url"]
    text = body["messages"][2]["content"]
    system_text = body["messages"][0]["content"]
    assert len(image_chunks) == 1
    assert "Setup plan:" in text
    assert "Current plan step:" in text
    assert "Wait until LibreOffice Writer is visible." in text
    assert "Task A microaction" not in system_text
    assert "Task A microaction" not in text
    assert json.loads(response)["tool_call"]["tool_name"] == "wait"
    assert cost_usd == 0.0


def test_hash_preflight_steps_changes_with_steps() -> None:
    steps_a = [{"op": "wait_for"}]
    steps_b = [{"op": "click"}]
    assert hash_preflight_steps(steps_a) != hash_preflight_steps(steps_b)


def test_select_manifest_entries_skip_and_limit() -> None:
    entries = discover_manifest_entries(TRACE_ROOT, TASK_EXAMPLES)
    settings = AugmentSettings(
        trace_root=TRACE_ROOT,
        task_examples_dir=TASK_EXAMPLES,
        mode="holo-native",
        skip_first_n=1,
        n_tasks=1,
    )
    selected = select_manifest_entries(entries, settings)
    assert len(selected) == 1
    assert selected[0].task_id == entries[1].task_id


def test_select_manifest_entries_by_task_ids() -> None:
    entries = discover_manifest_entries(TRACE_ROOT, TASK_EXAMPLES)
    settings = AugmentSettings(
        trace_root=TRACE_ROOT,
        task_examples_dir=TASK_EXAMPLES,
        mode="holo-native",
        task_ids=("task-b",),
    )
    selected = select_manifest_entries(entries, settings)
    assert len(selected) == 1
    assert selected[0].task_id == "task-b"


def test_skip_existing_augmented_task(tmp_path: Path, oracle_env: None) -> None:
    trace_root = tmp_path / "trace"
    trace_root.mkdir()
    for task_id in ("task-a",):
        src = TRACE_ROOT / "rollout/libreoffice_writer" / task_id
        dst = trace_root / "rollout/libreoffice_writer" / task_id
        dst.mkdir(parents=True, exist_ok=True)
        for path in src.iterdir():
            if path.is_file():
                dst.joinpath(path.name).write_bytes(path.read_bytes())

    settings = AugmentSettings(
        trace_root=trace_root,
        task_examples_dir=TASK_EXAMPLES,
        mode="openai-api",
        task_ids=("task-a",),
    )
    output_path = augmented_task_path(settings, "task-a")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("{}", encoding="utf-8")

    summary = run_augmentation(settings)
    assert summary["skipped"] == 1
    assert summary["results"][0]["status"] == "skipped"
    assert summary["results"][0]["source"] == "exists"
