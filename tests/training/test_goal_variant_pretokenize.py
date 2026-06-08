from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def import_training_module(name: str):
    sys.path.insert(0, str(REPO_ROOT / "training"))
    try:
        return __import__(name)
    finally:
        sys.path.pop(0)


def _make_processor(*, prefix_lens: list[int], full_len: int):
    processor = MagicMock()
    call_idx = {"n": 0}

    def apply_chat_template(messages, **kwargs):
        if kwargs.get("tokenize") is True and kwargs.get("return_dict"):
            return {"input_ids": torch.arange(full_len).unsqueeze(0)}
        if kwargs.get("tokenize") is False:
            return "x" * prefix_lens[min(call_idx["n"], len(prefix_lens) - 1)]
        call_idx["n"] += 1
        return {"input_ids": torch.arange(full_len).unsqueeze(0)}

    processor.apply_chat_template.side_effect = apply_chat_template
    processor.tokenizer.side_effect = lambda text, **kwargs: {
        "input_ids": list(range(len(text)))
    }
    processor.tokenizer.convert_tokens_to_ids.return_value = 99
    return processor


def test_expand_row_with_goal_variants_emits_one_record_per_variant():
    goal_variant_pretokenize = import_training_module("goal_variant_pretokenize")
    goal_variants = import_training_module("goal_variants")
    uitars_format = import_training_module("uitars_format")

    settings = uitars_format.UitarsFormatSettings()
    processor = _make_processor(prefix_lens=[3, 4], full_len=10)
    row = {
        "task_id": "test-task-001",
        "step_num": 4,
        "rollout_dir": "/tmp/rollout",
        "loss_on_last_assistant_only": True,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {"role": "user", "content": [{"type": "text", "text": "base instruction"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "target"}]},
        ],
    }
    variants = [
        goal_variants.GoalVariant(
            variant_index=0,
            goal="Goal A",
            expected_outcome="Outcome A",
            task_id="test-task-001",
            source_path=Path("goal_variant_0.json"),
        ),
        goal_variants.GoalVariant(
            variant_index=1,
            goal="Goal B",
            expected_outcome="Outcome B",
            task_id="test-task-001",
            source_path=Path("goal_variant_1.json"),
        ),
    ]
    caches = goal_variant_pretokenize.GoalVariantPretokenizeCaches()
    records = goal_variant_pretokenize.expand_row_with_goal_variants(
        processor,
        row,
        variants,
        settings,
        caches,
    )
    assert len(records) == 2
    assert records[0]["goal_variant_index"] == 0
    assert records[1]["goal_variant_index"] == 1
    assert len(caches.suffix_cache) == 1
    assert len(caches.prefix_cache) == 2


def test_expand_row_reuses_suffix_across_variants():
    goal_variant_pretokenize = import_training_module("goal_variant_pretokenize")
    goal_variants = import_training_module("goal_variants")
    uitars_format = import_training_module("uitars_format")

    settings = uitars_format.UitarsFormatSettings()
    processor = _make_processor(prefix_lens=[3, 5], full_len=12)
    row = {
        "task_id": "test-task-001",
        "step_num": 2,
        "rollout_dir": "/tmp/rollout",
        "loss_on_last_assistant_only": True,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {"role": "user", "content": [{"type": "text", "text": "base instruction"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "target"}]},
        ],
    }
    variants = [
        goal_variants.GoalVariant(
            variant_index=0,
            goal="Goal A",
            expected_outcome="Outcome A",
            task_id="test-task-001",
            source_path=Path("goal_variant_0.json"),
        ),
        goal_variants.GoalVariant(
            variant_index=1,
            goal="Goal B",
            expected_outcome="Outcome B",
            task_id="test-task-001",
            source_path=Path("goal_variant_1.json"),
        ),
    ]
    caches = goal_variant_pretokenize.GoalVariantPretokenizeCaches()
    records = goal_variant_pretokenize.expand_row_with_goal_variants(
        processor,
        row,
        variants,
        settings,
        caches,
    )
    suffix_key = ("/tmp/rollout", 2)
    shared_suffix = caches.suffix_cache[suffix_key].suffix_input_ids
    for record in records:
        input_ids = record["features"]["input_ids"]
        assert torch.equal(input_ids[-shared_suffix.numel() :], shared_suffix)
