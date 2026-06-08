from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class PretokenizedStats:
    actual_rows: int
    rollouts: int
    tasks: int
    goals: int | None
    shard_count: int


@dataclass(frozen=True)
class PretokenizedValidation:
    pretokenized_dir: Path
    staged_jsonl: Path
    expected_base_rows: int
    expected_training_rows: int
    stats: PretokenizedStats
    manifest: dict[str, Any] | None


def load_manifest(pretokenized_dir: Path) -> dict[str, Any] | None:
    path = pretokenized_dir / MANIFEST_FILENAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def write_manifest_file(output_dir: Path, payload: Mapping[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / MANIFEST_FILENAME
    path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
    return path


def count_jsonl_rows(data_path: Path) -> int:
    count = 0
    with open(data_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def load_jsonl_rows(data_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(data_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Each JSONL row must be an object: {data_path}")
                rows.append(payload)
    return rows


def list_shard_paths(pretokenized_dir: Path) -> list[Path]:
    return sorted(pretokenized_dir.glob("shard-*.pt"))


def load_shard_records(shard_path: Path) -> list[dict[str, Any]]:
    payload = torch.load(shard_path, weights_only=False)
    if not isinstance(payload, list):
        raise ValueError(f"Shard must contain a list: {shard_path}")
    records: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"Shard record must be an object: {shard_path}")
        features = item.get("features")
        if not isinstance(features, dict):
            raise ValueError(f"Shard record missing features dict: {shard_path}")
        for key in ("input_ids", "labels"):
            if key not in features:
                raise ValueError(f"Shard record missing features.{key}: {shard_path}")
        records.append(item)
    return records


def collect_pretokenized_stats(pretokenized_dir: Path) -> PretokenizedStats:
    shard_paths = list_shard_paths(pretokenized_dir)
    if not shard_paths:
        raise ValueError(f"No pretokenized shards found under {pretokenized_dir}")

    records: list[dict[str, Any]] = []
    for shard_path in shard_paths:
        shard_records = load_shard_records(shard_path)
        if not shard_records:
            raise ValueError(f"Empty pretokenized shard: {shard_path}")
        records.extend(shard_records)

    rollout_dirs = {
        str(record["rollout_dir"])
        for record in records
        if isinstance(record.get("rollout_dir"), str)
    }
    task_ids = {
        str(record["task_id"])
        for record in records
        if isinstance(record.get("task_id"), str)
    }
    goals = {
        str(record["goal"])
        for record in records
        if isinstance(record.get("goal"), str)
    }
    return PretokenizedStats(
        actual_rows=len(records),
        rollouts=len(rollout_dirs),
        tasks=len(task_ids),
        goals=len(goals) if goals else None,
        shard_count=len(shard_paths),
    )


def expected_training_rows_for_goal_variants(
    rows: Sequence[dict[str, Any]],
    catalog: Mapping[str, Sequence[Any]],
) -> int:
    expected = 0
    for row in rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str):
            continue
        variants = catalog.get(task_id.strip())
        if not variants:
            continue
        expected += len(variants)
    return expected


def validate_manifest_metadata(
    manifest: Mapping[str, Any],
    *,
    goal_variants: bool,
    max_goal_variants: int,
    task_generation_root: str | None,
    base_rows: int,
    expanded_rows: int,
    rollouts: int,
    tasks: int,
    shard_count: int,
    base_model: str | None = None,
) -> list[str]:
    errors: list[str] = []
    checks: list[tuple[str, Any, Any]] = [
        ("goal_variants", manifest.get("goal_variants"), goal_variants),
        ("max_goal_variants", manifest.get("max_goal_variants"), max_goal_variants),
        ("task_generation_root", manifest.get("task_generation_root"), task_generation_root),
        ("base_rows", manifest.get("base_rows"), base_rows),
        ("expanded_rows", manifest.get("expanded_rows"), expanded_rows),
        ("rollouts", manifest.get("rollouts"), rollouts),
        ("tasks", manifest.get("tasks"), tasks),
        ("shard_count", manifest.get("shard_count"), shard_count),
    ]
    if base_model is not None and "base_model" in manifest:
        checks.append(("base_model", manifest.get("base_model"), base_model))
    for field, actual, expected in checks:
        if field not in manifest:
            continue
        if actual != expected:
            errors.append(f"manifest.{field}={actual!r} does not match expected {expected!r}")
    return errors
