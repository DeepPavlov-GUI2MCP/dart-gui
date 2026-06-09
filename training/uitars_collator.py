from __future__ import annotations

import json
import os
import sys
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from tqdm import tqdm

from microaction_split import load_split, normalize_pretokenized_split, split_path
from uitars_format import UitarsFormatSettings, resolve_staged_messages


@dataclass(frozen=True)
class UitarsCollatorSettings:
    loss_on_last_assistant_only: bool = True


def find_assistant_indices(messages: Sequence[dict[str, Any]]) -> list[int]:
    return [idx for idx, message in enumerate(messages) if message.get("role") == "assistant"]


def image_token_lengths(processor: Any, full_inputs: dict[str, Any]) -> list[int]:
    image_grid_thw = full_inputs.get("image_grid_thw")
    if image_grid_thw is None:
        return []
    merge_size = getattr(getattr(processor, "image_processor", None), "merge_size", 2)
    merge_length = int(merge_size) ** 2
    return [int(grid.prod().item()) // merge_length for grid in image_grid_thw]


def chat_template_token_length(
    processor: Any,
    messages: Sequence[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    image_lengths: Sequence[int],
) -> int:
    text = processor.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    input_ids = processor.tokenizer(text, add_special_tokens=False)["input_ids"]
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    image_count = input_ids.count(image_token_id)
    return len(input_ids) + sum(image_lengths[:image_count]) - image_count


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
    image_lengths = image_token_lengths(processor, full_inputs)

    if loss_on_last_assistant_only:
        assistant_indices = find_assistant_indices(messages)
        if not assistant_indices:
            raise ValueError("Expected at least one assistant message.")
        last_assistant_idx = assistant_indices[-1]
        prefix_messages = list(messages[:last_assistant_idx])
        prefix_len = chat_template_token_length(
            processor,
            prefix_messages,
            add_generation_prompt=True,
            image_lengths=image_lengths,
        )
        labels[:prefix_len] = -100
    else:
        labels[:] = -100
        assistant_indices = find_assistant_indices(messages)
        for assistant_idx in assistant_indices:
            start_idx = assistant_idx
            end_idx = assistant_idx + 1
            prefix_messages = list(messages[:start_idx])
            if prefix_messages:
                prefix_len = chat_template_token_length(
                    processor,
                    prefix_messages,
                    add_generation_prompt=True,
                    image_lengths=image_lengths,
                )
            else:
                prefix_len = 0
            end_len = chat_template_token_length(
                processor,
                list(messages[:end_idx]),
                add_generation_prompt=False,
                image_lengths=image_lengths,
            )
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
        self.image_cache: dict[Path, dict[str, Any]] = {}
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
        messages = resolve_staged_messages(row["messages"], rollout_dir, self.settings, self.image_cache)
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


def _load_pretokenized_manifest(pretokenized_dir: Path) -> dict[str, Any]:
    path = pretokenized_dir / "manifest.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _shard_worker_key(path: Path) -> tuple[int, int] | None:
    parts = path.stem.split("-")
    if len(parts) != 3:
        return None
    try:
        return int(parts[1]), int(parts[2])
    except ValueError:
        return None


def _load_shard_records(shard_path: Path) -> list[dict[str, Any]]:
    payload = torch.load(shard_path, weights_only=False)
    if not isinstance(payload, list):
        raise ValueError(f"Pretokenized shard must contain a list: {shard_path}")
    return [item for item in payload if isinstance(item, dict)]


def _infer_chunked_shard_counts(
    shard_paths: Sequence[Path],
    *,
    total_records: int | None,
) -> list[int] | None:
    keyed_paths: list[tuple[int, int, Path]] = []
    for shard_path in shard_paths:
        key = _shard_worker_key(shard_path)
        if key is None:
            return None
        keyed_paths.append((key[0], key[1], shard_path))
    if not keyed_paths:
        return []

    first_count = len(_load_shard_records(keyed_paths[0][2]))
    by_worker: dict[int, list[tuple[int, Path]]] = {}
    for worker_id, chunk_idx, shard_path in keyed_paths:
        by_worker.setdefault(worker_id, []).append((chunk_idx, shard_path))

    counts_by_path: dict[Path, int] = {}
    for chunks in by_worker.values():
        chunks.sort()
        for _, shard_path in chunks[:-1]:
            counts_by_path[shard_path] = first_count
        last_path = chunks[-1][1]
        counts_by_path[last_path] = len(_load_shard_records(last_path))

    counts = [counts_by_path[path] for path in shard_paths]
    if total_records is not None and sum(counts) != total_records:
        raise ValueError(
            f"Inferred {sum(counts)} pretokenized records, but manifest expects {total_records}."
        )
    return counts


def _is_simple_worker_shard(path: Path) -> bool:
    parts = path.stem.split("-")
    return len(parts) == 2 and parts[0] == "shard"


def _infer_simple_shard_counts(
    shard_paths: Sequence[Path],
    *,
    total_records: int | None,
) -> list[int] | None:
    if not shard_paths or not all(_is_simple_worker_shard(path) for path in shard_paths):
        return None
    if total_records is None:
        return None
    if len(shard_paths) == 1:
        return [total_records]
    first_count = len(_load_shard_records(shard_paths[0]))
    counts = [first_count] * (len(shard_paths) - 1)
    last_count = total_records - first_count * (len(shard_paths) - 1)
    if last_count <= 0:
        last_count = len(_load_shard_records(shard_paths[-1]))
    counts.append(last_count)
    if sum(counts) != total_records:
        return None
    return counts


def _load_shard_counts(shard_paths: Sequence[Path], manifest: dict[str, Any]) -> list[int]:
    total_records = manifest.get("expanded_rows")
    if not isinstance(total_records, int):
        total_records = None
    if len(shard_paths) == 1 and total_records is not None:
        return [total_records]

    counts = _infer_chunked_shard_counts(shard_paths, total_records=total_records)
    if counts is not None:
        return counts

    counts = _infer_simple_shard_counts(shard_paths, total_records=total_records)
    if counts is not None:
        return counts

    counts = [len(_load_shard_records(path)) for path in tqdm(shard_paths, desc="counting shards", unit="shard")]
    if total_records is not None and sum(counts) != total_records:
        raise ValueError(
            f"Found {sum(counts)} pretokenized records, but manifest expects {total_records}."
        )
    return counts


class PretokenizedUitarsDataset(IterableDataset):
    def __init__(
        self,
        pretokenized_dir: Path,
        *,
        split: str = "all",
        smoke_microactions: int | None = None,
    ) -> None:
        self.pretokenized_dir = Path(pretokenized_dir)
        self.split = normalize_pretokenized_split(split)
        self.manifest = _load_pretokenized_manifest(self.pretokenized_dir)
        split_payload = load_split(self.pretokenized_dir, smoke_microactions=smoke_microactions)
        split = normalize_pretokenized_split(split)
        if split != "all":
            if split_payload is None:
                raise ValueError(
                    f"Pretokenized split {split!r} requested but "
                    f"{split_path(self.pretokenized_dir, smoke_microactions=smoke_microactions)} is missing."
                )
            self.allowed_task_ids = split_payload.task_ids_for(split)
            self.split_row_count = split_payload.row_count_for(
                split,
                total_rows=self.manifest.get("expanded_rows"),
            )
        else:
            self.allowed_task_ids = None
            self.split_row_count = None
        self.shard_paths = sorted(self.pretokenized_dir.glob("shard-*.pt"))
        if not self.shard_paths:
            raise ValueError(f"No pretokenized shards found under {pretokenized_dir}")
        self.shard_counts = _load_shard_counts(self.shard_paths, _load_pretokenized_manifest(self.pretokenized_dir))
        if sum(self.shard_counts) == 0:
            raise ValueError(f"No pretokenized records found under {pretokenized_dir}")
        self.cumulative_counts: list[int] = []
        running = 0
        for count in self.shard_counts:
            running += count
            self.cumulative_counts.append(running)
        self._cached_shard_path: Path | None = None
        self._cached_records: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return self._rank_length()

    def _rank_length(self) -> int:
        rank, world_size = self._rank_info()
        if self.split_row_count is not None:
            if world_size == 1:
                return self.split_row_count
            base = self.split_row_count // world_size
            return base + int(rank < self.split_row_count % world_size)
        records_per_rank = [
            sum(count for shard_idx, count in enumerate(self.shard_counts) if shard_idx % world_size == item_rank)
            for item_rank in range(world_size)
        ]
        return records_per_rank[0] if world_size == 1 else min(records_per_rank)

    @staticmethod
    def _worker_length(rank_length: int, worker_id: int, worker_count: int) -> int:
        base = rank_length // worker_count
        return base + int(worker_id < rank_length % worker_count)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= self.cumulative_counts[-1]:
            raise IndexError(index)
        shard_idx = bisect_right(self.cumulative_counts, index)
        previous_count = 0 if shard_idx == 0 else self.cumulative_counts[shard_idx - 1]
        record = self._load_cached_shard(self.shard_paths[shard_idx])[index - previous_count]
        if not self._record_matches_split(record):
            raise IndexError(index)
        return self._record_to_item(record)

    def __iter__(self):
        if self.allowed_task_ids is not None:
            yield from self._iter_record_sharded()
            return
        yield from self._iter_shard_sharded()

    def _iter_shard_sharded(self):
        rank, world_size = self._rank_info()
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        worker_length = self._worker_length(self._rank_length(), worker_id, worker_count)
        if worker_length == 0:
            return
        assigned = [
            idx
            for idx in range(len(self.shard_paths))
            if idx % world_size == rank and (idx // world_size) % worker_count == worker_id
        ]
        progress = tqdm(
            assigned,
            desc=f"rank {rank} streaming pretokenized shards",
            unit="shard",
            mininterval=1.0,
            file=sys.stderr,
        )
        records_seen = 0
        yielded = 0
        for shard_idx in progress:
            shard_path = self.shard_paths[shard_idx]
            records = _load_shard_records(shard_path)
            records_seen += len(records)
            progress.set_postfix(records=records_seen, last=shard_path.name, refresh=False)
            for record in records:
                if not self._record_matches_split(record):
                    continue
                yield self._record_to_item(record)
                yielded += 1
                if yielded >= worker_length:
                    return

    def _iter_record_sharded(self):
        rank, world_size = self._rank_info()
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        worker_length = self._worker_length(self._rank_length(), worker_id, worker_count)
        if worker_length == 0:
            return
        progress = tqdm(
            self.shard_paths,
            desc=f"rank {rank} streaming pretokenized shards",
            unit="shard",
            mininterval=1.0,
            file=sys.stderr,
        )
        filtered_index = 0
        rank_local_index = 0
        yielded = 0
        for shard_path in progress:
            records = _load_shard_records(shard_path)
            progress.set_postfix(records=filtered_index, last=shard_path.name, refresh=False)
            for record in records:
                if not self._record_matches_split(record):
                    continue
                if filtered_index % world_size != rank:
                    filtered_index += 1
                    continue
                if rank_local_index % worker_count != worker_id:
                    rank_local_index += 1
                    filtered_index += 1
                    continue
                yield self._record_to_item(record)
                yielded += 1
                rank_local_index += 1
                filtered_index += 1
                if yielded >= worker_length:
                    return

    def _record_matches_split(self, record: dict[str, Any]) -> bool:
        if self.allowed_task_ids is None:
            return True
        task_id = record.get("task_id")
        return isinstance(task_id, str) and task_id in self.allowed_task_ids

    @staticmethod
    def _rank_info() -> tuple[int, int]:
        rank = int(os.environ.get("RANK", "0"))
        world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
        return rank, world_size

    def _load_cached_shard(self, shard_path: Path) -> list[dict[str, Any]]:
        if self._cached_shard_path != shard_path:
            self._cached_records = _load_shard_records(shard_path)
            self._cached_shard_path = shard_path
        return self._cached_records

    @staticmethod
    def _record_to_item(record: dict[str, Any]) -> dict[str, Any]:
        features = record.get("features")
        if not isinstance(features, dict):
            raise ValueError("Pretokenized record is missing features.")
        return {"features": features}


def pad_feature_batch(
    features: list[dict[str, torch.Tensor]],
    *,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
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


def make_pretokenized_collator(pad_token_id: int) -> Callable[[list[dict[str, Any]]], dict[str, torch.Tensor]]:
    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        features = [item["features"] for item in batch]
        return pad_feature_batch(features, pad_token_id=pad_token_id)

    return collate
