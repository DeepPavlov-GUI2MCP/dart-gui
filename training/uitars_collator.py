from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from torch.utils.data import Dataset

from uitars_format import UitarsFormatSettings, resolve_staged_messages


@dataclass(frozen=True)
class UitarsCollatorSettings:
    loss_on_last_assistant_only: bool = True


def find_assistant_indices(messages: Sequence[dict[str, Any]]) -> list[int]:
    return [idx for idx, message in enumerate(messages) if message.get("role") == "assistant"]


def tokenize_uitars_messages(
    processor: Any,
    messages: Sequence[dict[str, Any]],
    *,
    loss_on_last_assistant_only: bool,
) -> dict[str, torch.Tensor]:
    full_inputs = processor.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = full_inputs["input_ids"][0]
    labels = input_ids.clone()

    if loss_on_last_assistant_only:
        assistant_indices = find_assistant_indices(messages)
        if not assistant_indices:
            raise ValueError("Expected at least one assistant message.")
        last_assistant_idx = assistant_indices[-1]
        prefix_messages = list(messages[:last_assistant_idx])
        prefix_inputs = processor.apply_chat_template(
            prefix_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        prefix_len = int(prefix_inputs["input_ids"].shape[-1])
        labels[:prefix_len] = -100
    else:
        labels[:] = -100
        assistant_indices = find_assistant_indices(messages)
        for assistant_idx in assistant_indices:
            start_idx = assistant_idx
            end_idx = assistant_idx + 1
            prefix_messages = list(messages[:start_idx])
            if prefix_messages:
                prefix_inputs = processor.apply_chat_template(
                    prefix_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                prefix_len = int(prefix_inputs["input_ids"].shape[-1])
            else:
                prefix_len = 0
            through_inputs = processor.apply_chat_template(
                list(messages[:end_idx]),
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
            end_len = int(through_inputs["input_ids"].shape[-1])
            labels[prefix_len:end_len] = input_ids[prefix_len:end_len]

    attention_mask = full_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask[0]

    batch: dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }
    for key, value in full_inputs.items():
        if key in {"input_ids", "attention_mask"}:
            continue
        if isinstance(value, torch.Tensor):
            if key in {"input_ids", "attention_mask", "mm_token_type_ids"} and value.ndim > 1:
                batch[key] = value[0]
            else:
                batch[key] = value
        else:
            batch[key] = value
    return batch


class UitarsTraceDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        settings: UitarsFormatSettings,
    ) -> None:
        self.settings = settings
        self.rows: list[dict[str, Any]] = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        rollout_dir = Path(row["rollout_dir"])
        messages = resolve_staged_messages(row["messages"], rollout_dir, self.settings)
        return {
            "messages": messages,
            "loss_on_last_assistant_only": bool(row.get("loss_on_last_assistant_only", True)),
        }


VISION_BATCH_KEYS = frozenset(
    {"pixel_values", "image_grid_thw", "video_grid_thw", "pixel_values_videos"}
)


def normalize_vision_feature(key: str, value: torch.Tensor) -> torch.Tensor:
    if key == "image_grid_thw" and value.ndim == 1 and value.numel() == 3:
        return value.unsqueeze(0)
    return value


def merge_vision_features(key: str, values: list[torch.Tensor]) -> torch.Tensor:
    normalized = [normalize_vision_feature(key, value) for value in values]
    if key in {"pixel_values", "image_grid_thw", "video_grid_thw", "pixel_values_videos"}:
        return torch.cat(normalized, dim=0)
    return normalized[0] if len(normalized) == 1 else torch.stack(normalized)


def make_uitars_data_collator(processor: Any) -> Callable[[list[dict[str, Any]]], dict[str, torch.Tensor]]:
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        features = [
            tokenize_uitars_messages(
                processor,
                item["messages"],
                loss_on_last_assistant_only=item["loss_on_last_assistant_only"],
            )
            for item in batch
        ]
        max_len = max(feature["input_ids"].shape[-1] for feature in features)
        batch_tensors: dict[str, list[torch.Tensor]] = {}
        for feature in features:
            seq_len = feature["input_ids"].shape[-1]
            pad_len = max_len - seq_len
            for key, value in feature.items():
                if key == "input_ids":
                    padded = torch.nn.functional.pad(value, (0, pad_len), value=pad_token_id)
                elif key == "labels":
                    padded = torch.nn.functional.pad(value, (0, pad_len), value=-100)
                elif key == "attention_mask":
                    padded = torch.nn.functional.pad(value, (0, pad_len), value=0)
                elif key in VISION_BATCH_KEYS:
                    padded = normalize_vision_feature(key, value)
                elif isinstance(value, torch.Tensor) and value.ndim == 1:
                    padded = torch.nn.functional.pad(value, (0, pad_len), value=0)
                else:
                    padded = value
                batch_tensors.setdefault(key, []).append(padded)
        merged: dict[str, torch.Tensor] = {}
        for key, values in batch_tensors.items():
            if key in VISION_BATCH_KEYS:
                merged[key] = merge_vision_features(key, values)
            else:
                merged[key] = torch.stack(values)
        return merged

    return collate
