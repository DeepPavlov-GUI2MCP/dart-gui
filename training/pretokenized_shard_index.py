from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch


SPLIT_SHARDS_FILENAME = "split-shards.json"
SPLIT_SHARDS_FILENAME_SMOKE = "split-smoke-shards.json"


def split_shards_filename(*, smoke_microactions: int | None = None) -> str:
    if smoke_microactions is not None:
        return SPLIT_SHARDS_FILENAME_SMOKE
    return SPLIT_SHARDS_FILENAME


def split_shards_path(pretokenized_dir: Path, *, smoke_microactions: int | None = None) -> Path:
    return Path(pretokenized_dir) / split_shards_filename(smoke_microactions=smoke_microactions)


def load_shard_records(shard_path: Path) -> list[dict[str, Any]]:
    payload = torch.load(shard_path, weights_only=False)
    if not isinstance(payload, list):
        raise ValueError(f"Pretokenized shard must contain a list: {shard_path}")
    return [item for item in payload if isinstance(item, dict)]


def _record_matches_split(record: Mapping[str, Any], allowed_task_ids: set[str]) -> bool:
    task_id = record.get("task_id")
    return isinstance(task_id, str) and task_id in allowed_task_ids


def compute_shard_index(
    shard_paths: Sequence[Path],
    *,
    allowed_task_ids: set[str],
    world_size: int,
    load_records: Callable[[Path], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    loader = load_records or load_shard_records
    shard_starts: dict[str, int] = {}
    rank_shards: list[list[str]] = [[] for _ in range(world_size)]
    rank_shard_seen: list[set[str]] = [set() for _ in range(world_size)]
    rank_rows = [0] * world_size
    filtered_index = 0

    for shard_path in shard_paths:
        shard_name = shard_path.name
        shard_starts[shard_name] = filtered_index
        for record in loader(shard_path):
            if not _record_matches_split(record, allowed_task_ids):
                continue
            rank = filtered_index % world_size
            if shard_name not in rank_shard_seen[rank]:
                rank_shard_seen[rank].add(shard_name)
                rank_shards[rank].append(shard_name)
            rank_rows[rank] += 1
            filtered_index += 1

    ranks_payload = [
        {"shards": shards, "rows": rows}
        for shards, rows in zip(rank_shards, rank_rows)
    ]
    return {
        "shard_starts": shard_starts,
        "filtered_rows": filtered_index,
        "ranks": ranks_payload,
    }


def compute_shard_indexes(
    shard_paths: Sequence[Path],
    *,
    split_task_ids: Mapping[str, set[str]],
    world_size: int,
    load_records: Callable[[Path], list[dict[str, Any]]] | None = None,
) -> dict[str, dict[str, Any]]:
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    loader = load_records or load_shard_records
    state: dict[str, dict[str, Any]] = {}
    for split_name in split_task_ids:
        state[split_name] = {
            "shard_starts": {},
            "rank_shards": [[] for _ in range(world_size)],
            "rank_shard_seen": [set() for _ in range(world_size)],
            "rank_rows": [0] * world_size,
            "filtered_index": 0,
        }

    for shard_path in shard_paths:
        shard_name = shard_path.name
        for split_state in state.values():
            split_state["shard_starts"][shard_name] = split_state["filtered_index"]

        for record in loader(shard_path):
            task_id = record.get("task_id")
            if not isinstance(task_id, str):
                continue
            for split_name, allowed_task_ids in split_task_ids.items():
                if task_id not in allowed_task_ids:
                    continue
                split_state = state[split_name]
                filtered_index = split_state["filtered_index"]
                rank = filtered_index % world_size
                rank_shard_seen = split_state["rank_shard_seen"][rank]
                if shard_name not in rank_shard_seen:
                    rank_shard_seen.add(shard_name)
                    split_state["rank_shards"][rank].append(shard_name)
                split_state["rank_rows"][rank] += 1
                split_state["filtered_index"] += 1

    indexes: dict[str, dict[str, Any]] = {}
    for split_name, split_state in state.items():
        indexes[split_name] = {
            "shard_starts": split_state["shard_starts"],
            "filtered_rows": split_state["filtered_index"],
            "ranks": [
                {"shards": shards, "rows": rows}
                for shards, rows in zip(split_state["rank_shards"], split_state["rank_rows"])
            ],
        }
    return indexes


def compute_shard_indexes_from_task_ids(
    shard_task_ids: Mapping[str, Sequence[str]],
    *,
    split_task_ids: Mapping[str, set[str]],
    world_size: int,
) -> dict[str, dict[str, Any]]:
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    state: dict[str, dict[str, Any]] = {}
    for split_name in split_task_ids:
        state[split_name] = {
            "shard_starts": {},
            "rank_shards": [[] for _ in range(world_size)],
            "rank_shard_seen": [set() for _ in range(world_size)],
            "rank_rows": [0] * world_size,
            "filtered_index": 0,
        }

    for shard_name, task_ids in shard_task_ids.items():
        for split_state in state.values():
            split_state["shard_starts"][shard_name] = split_state["filtered_index"]
        for task_id in task_ids:
            for split_name, allowed_task_ids in split_task_ids.items():
                if task_id not in allowed_task_ids:
                    continue
                split_state = state[split_name]
                filtered_index = split_state["filtered_index"]
                rank = filtered_index % world_size
                rank_shard_seen = split_state["rank_shard_seen"][rank]
                if shard_name not in rank_shard_seen:
                    rank_shard_seen.add(shard_name)
                    split_state["rank_shards"][rank].append(shard_name)
                split_state["rank_rows"][rank] += 1
                split_state["filtered_index"] += 1

    indexes: dict[str, dict[str, Any]] = {}
    for split_name, split_state in state.items():
        indexes[split_name] = {
            "shard_starts": split_state["shard_starts"],
            "filtered_rows": split_state["filtered_index"],
            "ranks": [
                {"shards": shards, "rows": rows}
                for shards, rows in zip(split_state["rank_shards"], split_state["rank_rows"])
            ],
        }
    return indexes


def compute_shard_local_indexes_from_task_ids(
    shard_task_ids: Mapping[str, Sequence[str]],
    *,
    split_task_ids: Mapping[str, set[str]],
    world_size: int,
) -> dict[str, dict[str, Any]]:
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    indexes: dict[str, dict[str, Any]] = {}
    for split_name, allowed_task_ids in split_task_ids.items():
        shard_counts = [
            (index, shard_name, sum(1 for task_id in task_ids if task_id in allowed_task_ids))
            for index, (shard_name, task_ids) in enumerate(shard_task_ids.items())
        ]
        filtered_rows = sum(count for _, _, count in shard_counts)
        base = filtered_rows // world_size
        targets = [base + int(rank < filtered_rows % world_size) for rank in range(world_size)]
        rank_items: list[list[tuple[int, str, int]]] = [[] for _ in range(world_size)]
        rank_rows = [0] * world_size
        remaining = list(targets)

        for index, shard_name, row_count in sorted(shard_counts, key=lambda item: item[2], reverse=True):
            if row_count == 0:
                continue
            candidates = [
                rank
                for rank, remaining_rows in enumerate(remaining)
                if row_count <= remaining_rows
            ]
            if candidates:
                rank = max(candidates, key=lambda item: remaining[item])
            else:
                rank = min(range(world_size), key=rank_rows.__getitem__)
            rank_items[rank].append((index, shard_name, row_count))
            rank_rows[rank] += row_count
            remaining[rank] -= row_count

        changed = True
        while changed:
            changed = False
            donors = [rank for rank in range(world_size) if rank_rows[rank] > targets[rank]]
            receivers = [rank for rank in range(world_size) if rank_rows[rank] < targets[rank]]
            for donor in donors:
                surplus = rank_rows[donor] - targets[donor]
                for receiver in receivers:
                    deficit = targets[receiver] - rank_rows[receiver]
                    needed = min(surplus, deficit)
                    item_index = next(
                        (
                            item_idx
                            for item_idx, (_, _, row_count) in enumerate(rank_items[donor])
                            if row_count == needed
                        ),
                        None,
                    )
                    if item_index is None:
                        continue
                    item = rank_items[donor].pop(item_index)
                    rank_items[receiver].append(item)
                    rank_rows[donor] -= item[2]
                    rank_rows[receiver] += item[2]
                    changed = True
                    break
                if changed:
                    break

        rank_shards = [
            [shard_name for _, shard_name, _ in sorted(items)]
            for items in rank_items
        ]
        indexes[split_name] = {
            "assignment": "shard_local",
            "filtered_rows": filtered_rows,
            "ranks": [
                {"shards": shards, "rows": rows}
                for shards, rows in zip(rank_shards, rank_rows)
            ],
        }
    return indexes


def build_shard_index_payload(
    pretokenized_dir: Path,
    *,
    split_name: str,
    allowed_task_ids: set[str],
    world_size: int,
    load_records: Callable[[Path], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    shard_paths = sorted(Path(pretokenized_dir).glob("shard-*.pt"))
    if not shard_paths:
        raise ValueError(f"No pretokenized shards found under {pretokenized_dir}")
    split_entry = compute_shard_index(
        shard_paths,
        allowed_task_ids=allowed_task_ids,
        world_size=world_size,
        load_records=load_records,
    )
    return {
        "split": split_name,
        "world_size": world_size,
        **split_entry,
    }


def load_shard_index(
    pretokenized_dir: Path,
    *,
    smoke_microactions: int | None = None,
) -> dict[str, Any] | None:
    path = split_shards_path(pretokenized_dir, smoke_microactions=smoke_microactions)
    if not path.is_file() and smoke_microactions is not None:
        legacy = split_shards_path(pretokenized_dir)
        if legacy.is_file():
            path = legacy
        else:
            return None
    elif not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def write_shard_index(
    pretokenized_dir: Path,
    payload: Mapping[str, Any],
    *,
    smoke_microactions: int | None = None,
) -> Path:
    path = split_shards_path(pretokenized_dir, smoke_microactions=smoke_microactions)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def get_split_world_entry(
    index: Mapping[str, Any],
    *,
    split_name: str,
    world_size: int,
) -> dict[str, Any] | None:
    splits = index.get("splits")
    if not isinstance(splits, dict):
        return None
    split_payload = splits.get(split_name)
    if not isinstance(split_payload, dict):
        return None
    world_entry = split_payload.get(str(world_size))
    if not isinstance(world_entry, dict):
        return None
    return world_entry


def ensure_shard_index(
    pretokenized_dir: Path,
    *,
    split_name: str,
    allowed_task_ids: set[str],
    world_size: int,
    smoke_microactions: int | None = None,
    load_records: Callable[[Path], list[dict[str, Any]]] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    index = load_shard_index(pretokenized_dir, smoke_microactions=smoke_microactions)
    if index is None:
        index = {"splits": {}}
    splits = dict(index.get("splits") or {})
    split_payload = dict(splits.get(split_name) or {})
    key = str(world_size)
    if not force and key in split_payload:
        return index
    expected = build_shard_index_payload(
        pretokenized_dir,
        split_name=split_name,
        allowed_task_ids=allowed_task_ids,
        world_size=world_size,
        load_records=load_records,
    )
    existing = split_payload.get(key)
    if existing == {
        "shard_starts": expected["shard_starts"],
        "filtered_rows": expected["filtered_rows"],
        "ranks": expected["ranks"],
    }:
        return index
    split_payload[key] = {
        "shard_starts": expected["shard_starts"],
        "filtered_rows": expected["filtered_rows"],
        "ranks": expected["ranks"],
    }
    splits[split_name] = split_payload
    index = {"splits": splits}
    write_shard_index(pretokenized_dir, index, smoke_microactions=smoke_microactions)
    return index


def ensure_shard_indexes(
    pretokenized_dir: Path,
    *,
    split_task_ids: Mapping[str, set[str]],
    world_size: int,
    smoke_microactions: int | None = None,
    load_records: Callable[[Path], list[dict[str, Any]]] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    index = load_shard_index(pretokenized_dir, smoke_microactions=smoke_microactions)
    if index is None:
        index = {"splits": {}}
    splits = dict(index.get("splits") or {})
    key = str(world_size)
    missing = {
        split_name: task_ids
        for split_name, task_ids in split_task_ids.items()
        if task_ids and (force or key not in dict(splits.get(split_name) or {}))
    }
    if not missing:
        return index

    shard_paths = sorted(Path(pretokenized_dir).glob("shard-*.pt"))
    if not shard_paths:
        raise ValueError(f"No pretokenized shards found under {pretokenized_dir}")
    built = compute_shard_indexes(
        shard_paths,
        split_task_ids=missing,
        world_size=world_size,
        load_records=load_records,
    )
    for split_name, split_entry in built.items():
        split_payload = dict(splits.get(split_name) or {})
        split_payload[key] = split_entry
        splits[split_name] = split_payload
    index = {"splits": splits}
    write_shard_index(pretokenized_dir, index, smoke_microactions=smoke_microactions)
    return index
