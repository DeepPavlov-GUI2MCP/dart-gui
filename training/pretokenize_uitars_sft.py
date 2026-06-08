from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from build_uitars_sft_dataset import (
    cache_name,
    load_model_settings,
    load_trace_scan_settings,
    pretokenized_cache_dir_name,
    stage_trace_dataset,
)
from pretokenized_manifest import write_manifest_file
from goal_variant_pretokenize import GoalVariantPretokenizeCaches, expand_row_with_goal_variants
from goal_variants import GoalVariant, assert_tasks_have_goal_variants, task_examples_path, task_generation_path
from uitars_collator import tokenize_uitars_messages
from uitars_format import UitarsFormatSettings, resolve_staged_messages
from uitars_trace_dataset import TraceScanSettings


LOGGER = logging.getLogger("pretokenize_uitars_sft")
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class WorkerSummary:
    worker_id: int
    rows: int
    expanded_rows: int
    input_tokens: int
    label_tokens: int
    processor_load_s: float
    resolve_s: float
    tokenize_s: float
    save_s: float
    output_path: str | None
    chunk_count: int


def shard_path_for_chunk(output_dir: Path, worker_id: int, chunk_idx: int, *, chunked: bool) -> Path:
    if chunked:
        return output_dir / f"shard-{worker_id:05d}-{chunk_idx:05d}.pt"
    return output_dir / f"shard-{worker_id:05d}.pt"


def save_record_chunk(
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    worker_id: int,
    chunk_idx: int,
    chunked: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_path_for_chunk(output_dir, worker_id, chunk_idx, chunked=chunked)
    torch.save(records, shard_path)
    return shard_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretokenize UI-TARS trace SFT rows in parallel.")
    parser.add_argument("--config", required=True, help="Path to train_sft YAML config.")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes.")
    parser.add_argument("--output-dir", default=None, help="Directory for pretokenized shards.")
    parser.add_argument("--refresh", action="store_true", help="Refresh staged JSONL before pretokenizing.")
    parser.add_argument("--max-rows", type=int, default=None, help="Only process the first N selected rows.")
    parser.add_argument(
        "--sort-by-cost",
        action="store_true",
        help="Process the largest image/text rows first; useful for throughput benchmarking.",
    )
    parser.add_argument(
        "--partition-mode",
        choices=("steps", "rollouts"),
        default="steps",
        help="Shard rows by per-step budget (default) or keep whole rollouts together.",
    )
    parser.add_argument("--no-save", action="store_true", help="Benchmark tokenization without writing shards.")
    parser.add_argument(
        "--flush-every",
        type=int,
        default=32,
        help="Write and clear in-memory records every N expanded rows (0 = one shard per worker).",
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Allow Hugging Face network checks while loading the processor.",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args(list(sys.argv[1:] if argv is None else argv))


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(processName)s %(message)s",
    )
    for noisy_logger in ("httpx", "httpcore", "huggingface_hub", "transformers"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def row_cost(row: dict[str, Any]) -> tuple[int, int]:
    images = 0
    chars = 0
    for message in row["messages"]:
        for item in message.get("content", []):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image":
                images += 1
            elif item.get("type") == "text":
                chars += len(item.get("text") or "")
    return images, chars


def load_rows(data_path: Path, *, max_rows: int | None, sort_by_cost: bool) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with open(data_path, encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if line:
                rows.append((idx, json.loads(line)))
    if sort_by_cost:
        rows.sort(key=lambda item: row_cost(item[1]), reverse=True)
    if max_rows is not None:
        rows = rows[:max_rows]
    return rows


def row_budget(row: dict[str, Any]) -> int:
    images, chars = row_cost(row)
    return images * 1_000_000 + chars


def partition_rows(
    indexed_rows: Sequence[tuple[int, dict[str, Any]]],
    workers: int,
    *,
    mode: str = "steps",
) -> list[list[tuple[int, dict[str, Any]]]]:
    if workers <= 1:
        return [list(indexed_rows)]

    if mode == "rollouts":
        groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for item in indexed_rows:
            groups[item[1]["rollout_dir"]].append(item)
        shards: list[list[tuple[int, dict[str, Any]]]] = [[] for _ in range(workers)]
        shard_budgets = [0 for _ in range(workers)]
        for group in sorted(groups.values(), key=lambda items: sum(row_budget(row) for _, row in items), reverse=True):
            shard_idx = min(range(workers), key=shard_budgets.__getitem__)
            shards[shard_idx].extend(group)
            shard_budgets[shard_idx] += sum(row_budget(row) for _, row in group)
        return shards

    shards = [[] for _ in range(workers)]
    shard_budgets = [0 for _ in range(workers)]
    for item in sorted(indexed_rows, key=lambda entry: (-row_budget(entry[1]), entry[0])):
        shard_idx = min(range(workers), key=shard_budgets.__getitem__)
        shards[shard_idx].append(item)
        shard_budgets[shard_idx] += row_budget(item[1])
    return shards


def serialize_goal_catalog(catalog: dict[str, list[GoalVariant]]) -> dict[str, list[dict[str, Any]]]:
    return {
        task_id: [
            {
                "variant_index": variant.variant_index,
                "goal": variant.goal,
                "expected_outcome": variant.expected_outcome,
                "task_id": variant.task_id,
                "source_path": str(variant.source_path),
                "task_index": variant.task_index,
            }
            for variant in variants
        ]
        for task_id, variants in catalog.items()
    }


def deserialize_goal_catalog(payload: dict[str, list[dict[str, Any]]]) -> dict[str, list[GoalVariant]]:
    catalog: dict[str, list[GoalVariant]] = {}
    for task_id, variants in payload.items():
        catalog[task_id] = [
            GoalVariant(
                variant_index=int(item["variant_index"]),
                goal=str(item["goal"]),
                expected_outcome=str(item["expected_outcome"]),
                task_id=str(item["task_id"]),
                source_path=Path(str(item["source_path"])),
                task_index=item.get("task_index"),
            )
            for item in variants
        ]
    return catalog


def pretokenize_worker(
    worker_id: int,
    rows: list[tuple[int, dict[str, Any]]],
    settings: UitarsFormatSettings,
    base_model: str,
    trust_remote_code: bool,
    local_files_only: bool,
    output_dir: str | None,
    no_save: bool,
    flush_every: int,
    *,
    goal_catalog: dict[str, list[dict[str, Any]]] | None,
) -> WorkerSummary:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoProcessor

    prep_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        base_model,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    processor_load_s = time.perf_counter() - prep_start

    image_cache: dict[Path, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    input_tokens = 0
    label_tokens = 0
    resolve_s = 0.0
    tokenize_s = 0.0
    save_s = 0.0
    expanded_rows = 0
    chunk_idx = 0
    chunk_count = 0
    output_path: str | None = None
    chunked_save = flush_every > 0
    flush_threshold = flush_every if chunked_save else 0
    output = Path(output_dir) if output_dir is not None else None
    variant_catalog = deserialize_goal_catalog(goal_catalog) if goal_catalog else None
    variant_caches = GoalVariantPretokenizeCaches() if variant_catalog else None

    def flush_records(*, final: bool = False) -> None:
        nonlocal chunk_idx, chunk_count, save_s, output_path
        if no_save or output is None or not records:
            records.clear()
            return
        if not final:
            if not chunked_save or len(records) < flush_threshold:
                return
        save_start = time.perf_counter()
        shard_path = save_record_chunk(
            records,
            output,
            worker_id=worker_id,
            chunk_idx=chunk_idx,
            chunked=chunked_save,
        )
        save_s += time.perf_counter() - save_start
        records.clear()
        output_path = str(shard_path)
        chunk_count += 1
        chunk_idx += 1

    for row_index, row in rows:
        rollout_dir = Path(row["rollout_dir"])
        if variant_catalog is not None and variant_caches is not None:
            task_id = row.get("task_id")
            if not isinstance(task_id, str) or task_id not in variant_catalog:
                continue
            resolve_start = time.perf_counter()
            expanded = expand_row_with_goal_variants(
                processor,
                row,
                variant_catalog[task_id],
                settings,
                variant_caches,
            )
            resolve_s += time.perf_counter() - resolve_start
            tokenize_s += 0.0
            for record in expanded:
                batch = record["features"]
                input_tokens += int(batch["input_ids"].numel())
                label_tokens += int((batch["labels"] != -100).sum().item())
                if not no_save:
                    records.append({"row_index": row_index, **record})
                expanded_rows += 1
                if not no_save:
                    flush_records()
            continue

        resolve_start = time.perf_counter()
        messages = resolve_staged_messages(row["messages"], rollout_dir, settings, image_cache)
        resolve_s += time.perf_counter() - resolve_start

        tokenize_start = time.perf_counter()
        batch = tokenize_uitars_messages(
            processor,
            messages,
            loss_on_last_assistant_only=bool(row.get("loss_on_last_assistant_only", True)),
        )
        tokenize_s += time.perf_counter() - tokenize_start

        input_tokens += int(batch["input_ids"].numel())
        label_tokens += int((batch["labels"] != -100).sum().item())
        expanded_rows += 1
        if not no_save:
            records.append(
                {
                    "row_index": row_index,
                    "task_id": row.get("task_id"),
                    "step_num": row.get("step_num"),
                    "rollout_dir": row.get("rollout_dir"),
                    "features": batch,
                }
            )
            flush_records()

    flush_records(final=True)

    return WorkerSummary(
        worker_id=worker_id,
        rows=len(rows),
        expanded_rows=expanded_rows,
        input_tokens=input_tokens,
        label_tokens=label_tokens,
        processor_load_s=processor_load_s,
        resolve_s=resolve_s,
        tokenize_s=tokenize_s,
        save_s=save_s,
        output_path=output_path,
        chunk_count=chunk_count,
    )


def write_manifest(
    output_dir: Path,
    *,
    settings: TraceScanSettings,
    base_model: str,
    base_rows: int,
    expanded_rows: int,
    task_count: int,
    rollout_count: int,
    shard_count: int,
    variants_per_task: int,
) -> None:
    write_manifest_file(
        output_dir,
        {
            "base_model": base_model,
            "goal_variants": settings.goal_variants,
            "max_goal_variants": settings.max_goal_variants,
            "task_generation_root": settings.task_generation_root,
            "base_rows": base_rows,
            "expanded_rows": expanded_rows,
            "tasks": task_count,
            "rollouts": rollout_count,
            "shard_count": shard_count,
            "variants_per_task": variants_per_task,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    total_start = time.perf_counter()

    LOGGER.info("[preparation] loading config")
    config_start = time.perf_counter()
    settings, datasets_dir, config_refresh = load_trace_scan_settings(args.config)
    base_model, trust_remote_code = load_model_settings(args.config)
    LOGGER.info("[preparation] config loaded in %.3fs", time.perf_counter() - config_start)

    LOGGER.info("[preparation] staging dataset if needed")
    stage_start = time.perf_counter()
    data_path = stage_trace_dataset(settings, datasets_dir, refresh=config_refresh or args.refresh)
    LOGGER.info("[preparation] staged data path=%s in %.3fs", data_path, time.perf_counter() - stage_start)

    LOGGER.info("[preparation] loading staged rows")
    load_start = time.perf_counter()
    indexed_rows = load_rows(data_path, max_rows=args.max_rows, sort_by_cost=args.sort_by_cost)
    LOGGER.info("[preparation] loaded rows=%d in %.3fs", len(indexed_rows), time.perf_counter() - load_start)

    goal_catalog_payload: dict[str, list[dict[str, Any]]] | None = None
    if settings.goal_variants:
        if not settings.task_generation_root:
            raise ValueError("`dataset.task_generation_root` is required when goal_variants is enabled.")
        rows_only = [row for _, row in indexed_rows]
        catalog = assert_tasks_have_goal_variants(
            task_generation_path(settings.task_generation_root),
            task_examples_path(settings),
            rows_only,
            max_variants=settings.max_goal_variants,
        )
        goal_catalog_payload = serialize_goal_catalog(catalog)
        LOGGER.info(
            "[preparation] goal variants enabled tasks=%d variants_per_task=%d",
            len(catalog),
            settings.max_goal_variants,
        )

    worker_count = max(1, min(args.workers, len(indexed_rows)))
    LOGGER.info("[preparation] partitioning rows workers=%d mode=%s", worker_count, args.partition_mode)
    partition_start = time.perf_counter()
    shards = partition_rows(indexed_rows, worker_count, mode=args.partition_mode)
    LOGGER.info(
        "[preparation] partitioned rows shard_sizes=%s in %.3fs",
        [len(shard) for shard in shards],
        time.perf_counter() - partition_start,
    )

    output_dir = args.output_dir
    if output_dir is None and not args.no_save:
        output_dir = str(data_path.parent / pretokenized_cache_dir_name(settings))

    LOGGER.info(
        "[data] starting workers no_save=%s flush_every=%d output_dir=%s",
        args.no_save,
        args.flush_every,
        output_dir,
    )
    data_start = time.perf_counter()
    summaries: list[WorkerSummary] = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                pretokenize_worker,
                worker_id,
                shard,
                settings.uitars,
                base_model,
                trust_remote_code,
                not args.allow_remote,
                output_dir,
                args.no_save,
                args.flush_every,
                goal_catalog=goal_catalog_payload,
            )
            for worker_id, shard in enumerate(shards)
            if shard
        ]
        for future in as_completed(futures):
            summary = future.result()
            summaries.append(summary)
            LOGGER.info(
                "[data] worker=%d rows=%d expanded_rows=%d chunks=%d prep.processor_load_s=%.3f "
                "data.resolve_s=%.3f data.tokenize_s=%.3f data.save_s=%.3f input_tokens=%d "
                "label_tokens=%d output=%s",
                summary.worker_id,
                summary.rows,
                summary.expanded_rows,
                summary.chunk_count,
                summary.processor_load_s,
                summary.resolve_s,
                summary.tokenize_s,
                summary.save_s,
                summary.input_tokens,
                summary.label_tokens,
                summary.output_path,
            )

    data_s = time.perf_counter() - data_start
    base_rows = sum(summary.rows for summary in summaries)
    expanded_rows = sum(summary.expanded_rows for summary in summaries)
    input_tokens = sum(summary.input_tokens for summary in summaries)
    label_tokens = sum(summary.label_tokens for summary in summaries)
    processor_load_s = sum(summary.processor_load_s for summary in summaries)
    resolve_s = sum(summary.resolve_s for summary in summaries)
    tokenize_s = sum(summary.tokenize_s for summary in summaries)
    save_s = sum(summary.save_s for summary in summaries)

    if output_dir is not None and not args.no_save:
        rollout_dirs = {
            row["rollout_dir"]
            for _, row in indexed_rows
            if isinstance(row.get("rollout_dir"), str)
        }
        task_ids = {
            row["task_id"]
            for _, row in indexed_rows
            if isinstance(row.get("task_id"), str)
        }
        shard_paths = sorted(Path(output_dir).glob("shard-*.pt"))
        write_manifest(
            Path(output_dir),
            settings=settings,
            base_model=base_model,
            base_rows=base_rows,
            expanded_rows=expanded_rows,
            task_count=len(task_ids) if not settings.goal_variants else len(goal_catalog_payload or {}),
            rollout_count=len(rollout_dirs),
            shard_count=len(shard_paths),
            variants_per_task=settings.max_goal_variants,
        )

    LOGGER.info(
        "[summary] base_rows=%d expanded_rows=%d workers=%d wall_s=%.3f preparation_plus_data_s=%.3f "
        "worker_prep.processor_load_s=%.3f worker_data.resolve_s=%.3f "
        "worker_data.tokenize_s=%.3f worker_data.save_s=%.3f input_tokens=%d "
        "label_tokens=%d input_tokens_per_wall_s=%.1f",
        base_rows,
        expanded_rows,
        worker_count,
        data_s,
        time.perf_counter() - total_start,
        processor_load_s,
        resolve_s,
        tokenize_s,
        save_s,
        input_tokens,
        label_tokens,
        input_tokens / data_s if data_s else 0.0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
