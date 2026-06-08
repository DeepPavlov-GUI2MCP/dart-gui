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
SFT_HOLO_FIXTURE_CONFIG = REPO_ROOT / "training" / "configs" / "sft_holo_fixture.yml"
SFT_GOAL_VARIANTS_CONFIG = REPO_ROOT / "training" / "configs" / "sft_holo_goal_variants_fixture.yml"
FIXTURE_ROLLOUT = (
    REPO_ROOT
    / "tests/training/fixtures/uitars_trace/rollout/libreoffice_writer/test-task-id"
)
HOLO_FIXTURE_ROLLOUT = (
    REPO_ROOT
    / "tests/training/fixtures/holo_trace/rollout/libreoffice_writer/test-task-id"
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
        trace_source="uitars",
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


def test_holo_to_uitars_converter():
    holo_to_uitars = import_training_module("holo_to_uitars")
    click = holo_to_uitars.convert_holo_response(
        '{"thought":"Open menu","tool_call":{"tool_name":"click","element":"Help","x":100,"y":200,"button":"left"}}'
    )
    assert click == "Thought: Open menu\nAction: click(start_box='<|box_start|>(100,200)<|box_end|>')"

    write = holo_to_uitars.convert_holo_response(
        '{"thought":"Type text","tool_call":{"tool_name":"write","content":"hello","press_enter":false,"overwrite":false}}'
    )
    assert write == "Thought: Type text\nAction: type(content='hello')"

    answer = holo_to_uitars.convert_holo_response(
        '{"thought":"Done","tool_call":{"tool_name":"answer","content":"done"}}'
    )
    assert answer == "Thought: Done\nAction: finished(content='done')"

    fail = holo_to_uitars.convert_holo_response(
        '{"thought":"Blocked","tool_call":{"tool_name":"fail","reason":"cannot continue"}}'
    )
    assert fail == "Thought: Blocked\nAction: call_user()"


def test_uitars_messages_normalize_all_assistant_actions():
    uitars_format = import_training_module("uitars_format")
    image_paths = [
        HOLO_FIXTURE_ROLLOUT / "step_3_20250101@120002.png",
        HOLO_FIXTURE_ROLLOUT / "step_4_20250101@120003.png",
    ]
    messages = uitars_format.build_messages_from_images_and_responses(
        "Test task",
        image_paths,
        ["Thought: Open menu\nAction: click(start_box='(100,200)')"],
        uitars_format.UitarsFormatSettings(),
        target_response="Thought: Choose item\nAction: click(start_box='(300,400)')",
        stage_relative_paths=True,
        rollout_dir=HOLO_FIXTURE_ROLLOUT,
    )
    assistant_texts = [
        message["content"][0]["text"]
        for message in messages
        if message["role"] == "assistant"
    ]
    assert len(assistant_texts) == 2
    assert all("<|box_start|>" in text and "<|box_end|>" in text for text in assistant_texts)
    assert all("start_box='(" not in text for text in assistant_texts)


def test_build_holo_per_step_rows_use_prior_screenshot():
    uitars_trace_dataset = import_training_module("uitars_trace_dataset")
    uitars_format = import_training_module("uitars_format")
    settings = uitars_trace_dataset.TraceScanSettings(
        trace_roots=(str(REPO_ROOT / "tests/training/fixtures/holo_trace/rollout"),),
        task_examples_dir=str(REPO_ROOT / "tests/training/fixtures/holo_trace/task_examples"),
        sample_mode="per_step",
        trace_source="holo",
        uitars=uitars_format.UitarsFormatSettings(),
    )
    rows = uitars_trace_dataset.build_rows_for_rollout(HOLO_FIXTURE_ROLLOUT, settings)
    assert len(rows) == 2
    first_image = rows[0]["messages"][2]["content"][0]["path"]
    assert first_image == "step_3_20250101@120002.png"
    assert rows[0]["step_num"] == 4


def test_collator_masks_labels_outside_last_assistant():
    import torch

    uitars_collator = import_training_module("uitars_collator")
    processor = MagicMock()

    def apply_chat_template(messages, **kwargs):
        if kwargs.get("tokenize") is True:
            return {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7]])}
        return "x" * (4 if len(messages) <= 2 else 7)

    processor.apply_chat_template.side_effect = apply_chat_template
    processor.tokenizer.side_effect = lambda text, **kwargs: {
        "input_ids": list(range(len(text)))
    }
    processor.tokenizer.convert_tokens_to_ids.return_value = 99
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


def test_resolve_goal_variants_config_from_fixture():
    run_sft = import_training_module("run_sft")
    config = run_sft.load_config_file(str(SFT_GOAL_VARIANTS_CONFIG))
    assert config.dataset.goal_variants is True
    assert config.dataset.max_goal_variants == 1
    assert config.dataset.task_generation_root.endswith(
        "tests/training/fixtures/goal_variants/task_generation"
    )
    assert config.dataset.pretokenized_traces_dir.endswith(
        "tests/training/fixtures/pretokenized_shards"
    )


def test_dry_run_subprocess_holo_fixture():
    result = subprocess.run(
        [
            sys.executable,
            str(RUN_SFT),
            "--config",
            str(SFT_HOLO_FIXTURE_CONFIG),
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
