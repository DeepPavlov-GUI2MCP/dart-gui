from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from goal_variants import GoalVariant, replace_instruction_in_messages
from uitars_collator import (
    VISION_BATCH_KEYS,
    chat_template_token_length,
    find_assistant_indices,
    image_token_lengths,
    tokenize_uitars_messages,
)
from uitars_format import UitarsFormatSettings, resolve_staged_messages


@dataclass
class PrefixCacheEntry:
    input_ids: torch.Tensor
    prefix_len: int


@dataclass
class SuffixCacheEntry:
    suffix_input_ids: torch.Tensor
    shared_features: dict[str, Any]
    image_lengths: list[int]
    reference_prefix_len: int


@dataclass
class GoalVariantPretokenizeCaches:
    suffix_cache: dict[tuple[str, int], SuffixCacheEntry] = field(default_factory=dict)
    prefix_cache: dict[tuple[str, int], PrefixCacheEntry] = field(default_factory=dict)
    image_cache: dict[Path, dict[str, Any]] = field(default_factory=dict)


def _prefix_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(messages) < 2:
        raise ValueError("Expected at least system and instruction user messages.")
    return list(messages[:2])


def tokenize_prefix(
    processor: Any,
    prefix_messages: list[dict[str, Any]],
) -> PrefixCacheEntry:
    full_inputs = processor.apply_chat_template(
        prefix_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = full_inputs["input_ids"][0]
    prefix_len = int(input_ids.numel())
    return PrefixCacheEntry(input_ids=input_ids, prefix_len=prefix_len)


def build_suffix_cache_entry(
    processor: Any,
    resolved_messages: list[dict[str, Any]],
    *,
    loss_on_last_assistant_only: bool,
) -> SuffixCacheEntry:
    ref_batch = tokenize_uitars_messages(
        processor,
        resolved_messages,
        loss_on_last_assistant_only=loss_on_last_assistant_only,
    )
    image_lengths = image_token_lengths(processor, ref_batch)
    reference_prefix_len = chat_template_token_length(
        processor,
        _prefix_messages(resolved_messages),
        add_generation_prompt=True,
        image_lengths=[],
    )
    suffix_input_ids = ref_batch["input_ids"][reference_prefix_len:].clone()
    shared_features = {
        key: value
        for key, value in ref_batch.items()
        if key not in {"input_ids", "labels", "attention_mask"}
    }
    return SuffixCacheEntry(
        suffix_input_ids=suffix_input_ids,
        shared_features=shared_features,
        image_lengths=image_lengths,
        reference_prefix_len=reference_prefix_len,
    )


def assemble_variant_features(
    processor: Any,
    prefix_entry: PrefixCacheEntry,
    suffix_entry: SuffixCacheEntry,
    variant_messages: list[dict[str, Any]],
    *,
    loss_on_last_assistant_only: bool,
) -> dict[str, torch.Tensor]:
    input_ids = torch.cat([prefix_entry.input_ids, suffix_entry.suffix_input_ids], dim=0)
    labels = input_ids.clone()
    if loss_on_last_assistant_only:
        assistant_indices = find_assistant_indices(variant_messages)
        if not assistant_indices:
            raise ValueError("Expected at least one assistant message.")
        prefix_messages = list(variant_messages[: assistant_indices[-1]])
        mask_len = chat_template_token_length(
            processor,
            prefix_messages,
            add_generation_prompt=True,
            image_lengths=suffix_entry.image_lengths,
        )
        labels[:mask_len] = -100
    else:
        labels[:] = -100
        assistant_indices = find_assistant_indices(variant_messages)
        for assistant_idx in assistant_indices:
            prefix_messages = list(variant_messages[:assistant_idx])
            if prefix_messages:
                prefix_len = chat_template_token_length(
                    processor,
                    prefix_messages,
                    add_generation_prompt=True,
                    image_lengths=suffix_entry.image_lengths,
                )
            else:
                prefix_len = 0
            end_len = chat_template_token_length(
                processor,
                list(variant_messages[: assistant_idx + 1]),
                add_generation_prompt=False,
                image_lengths=suffix_entry.image_lengths,
            )
            labels[prefix_len:end_len] = input_ids[prefix_len:end_len]

    attention_mask = torch.ones_like(input_ids)
    batch: dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }
    for key, value in suffix_entry.shared_features.items():
        if key in VISION_BATCH_KEYS and isinstance(value, torch.Tensor):
            batch[key] = value
    return batch


def expand_row_with_goal_variants(
    processor: Any,
    row: dict[str, Any],
    variants: list[GoalVariant],
    settings: UitarsFormatSettings,
    caches: GoalVariantPretokenizeCaches,
) -> list[dict[str, Any]]:
    rollout_dir = Path(row["rollout_dir"])
    step_num = int(row.get("step_num", -1))
    task_id = str(row.get("task_id") or "")
    loss_on_last_assistant_only = bool(row.get("loss_on_last_assistant_only", True))
    suffix_key = (str(rollout_dir), step_num)

    if suffix_key not in caches.suffix_cache:
        resolved_ref = resolve_staged_messages(
            row["messages"],
            rollout_dir,
            settings,
            caches.image_cache,
        )
        caches.suffix_cache[suffix_key] = build_suffix_cache_entry(
            processor,
            resolved_ref,
            loss_on_last_assistant_only=loss_on_last_assistant_only,
        )

    suffix_entry = caches.suffix_cache[suffix_key]
    records: list[dict[str, Any]] = []
    for variant in variants:
        prefix_key = (task_id, variant.variant_index)
        if prefix_key not in caches.prefix_cache:
            variant_prefix_messages = replace_instruction_in_messages(
                row["messages"],
                variant.goal,
                settings,
            )
            resolved_prefix = resolve_staged_messages(
                _prefix_messages(variant_prefix_messages),
                rollout_dir,
                settings,
                caches.image_cache,
            )
            caches.prefix_cache[prefix_key] = tokenize_prefix(processor, resolved_prefix)

        variant_messages = replace_instruction_in_messages(
            row["messages"],
            variant.goal,
            settings,
        )
        resolved_variant = resolve_staged_messages(
            variant_messages,
            rollout_dir,
            settings,
            caches.image_cache,
        )
        features = assemble_variant_features(
            processor,
            caches.prefix_cache[prefix_key],
            suffix_entry,
            resolved_variant,
            loss_on_last_assistant_only=loss_on_last_assistant_only,
        )
        records.append(
            {
                "task_id": task_id,
                "step_num": step_num,
                "rollout_dir": str(rollout_dir),
                "goal_variant_index": variant.variant_index,
                "instruction": variant.goal,
                "features": features,
            }
        )
    return records
