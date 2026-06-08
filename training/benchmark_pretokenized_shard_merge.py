#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from uitars_collator import _load_shard_records


def pick_shards(pretokenized_dir: Path, count: int, *, selection: str) -> list[Path]:
    shards = list(pretokenized_dir.glob("shard-*.pt"))
    if len(shards) < count:
        raise ValueError(f"Need at least {count} shards under {pretokenized_dir}")
    if selection == "smallest":
        return sorted(shards, key=lambda p: p.stat().st_size)[:count]
    if selection == "median":
        median_size = statistics.median(path.stat().st_size for path in shards)
        return [
            path
            for path, _ in sorted(
                ((path, path.stat().st_size) for path in shards),
                key=lambda item: abs(item[1] - median_size),
            )[:count]
        ]
    raise ValueError(f"Unsupported selection: {selection}")


def build_merge_layouts(source_shards: list[Path], out_dir: Path) -> dict[int, list[Path]]:
    if len(source_shards) % 8 != 0 and len(source_shards) % 4 != 0:
        raise ValueError("shard-count must be divisible by 4")
    out_dir.mkdir(parents=True, exist_ok=True)
    records_by_shard = [_load_shard_records(path) for path in source_shards]
    total_records = sum(len(records) for records in records_by_shard)
    layouts: dict[int, list[Path]] = {1: list(source_shards)}

    for originals_per_file in (2, 4, 8):
        if len(source_shards) % originals_per_file != 0:
            continue
        merged_paths: list[Path] = []
        for file_idx in range(len(source_shards) // originals_per_file):
            start = file_idx * originals_per_file
            end = start + originals_per_file
            merged_records: list[dict] = []
            for records in records_by_shard[start:end]:
                merged_records.extend(records)
            merged_path = out_dir / f"merged-{originals_per_file}x-{file_idx:02d}.pt"
            torch.save(merged_records, merged_path)
            merged_paths.append(merged_path)
        if sum(len(_load_shard_records(path)) for path in merged_paths) != total_records:
            raise ValueError(f"Merged shard record count mismatch for {originals_per_file}x")
        layouts[originals_per_file] = merged_paths
    return layouts


def load_paths(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        total += len(_load_shard_records(path))
    return total


def benchmark_paths(paths: list[Path], *, repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        load_paths(paths)
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        record_count = load_paths(paths)
        timings.append(time.perf_counter() - start)
        if record_count <= 0:
            raise ValueError("Expected records while benchmarking")
    return timings


def summarize(label: str, timings: list[float], *, file_count: int, total_bytes: int) -> dict:
    stats = {
        "label": label,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "repeats": len(timings),
        "min_s": min(timings),
        "median_s": statistics.median(timings),
        "mean_s": statistics.mean(timings),
        "max_s": max(timings),
    }
    print(
        f"{label}: files={file_count} size_gb={total_bytes / (1024**3):.3f} "
        f"min={stats['min_s']:.3f}s median={stats['median_s']:.3f}s mean={stats['mean_s']:.3f}s"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretokenized-dir",
        type=Path,
        default=Path(
            "datasets/runtime/uitars-traces-b8a44e9156a4/pretokenized-uitars-traces-b8a44e9156a4"
        ),
    )
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--selection", choices=("smallest", "median"), default="median")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("datasets/runtime/benchmark-shard-merge-median8"),
    )
    args = parser.parse_args()

    source_shards = pick_shards(args.pretokenized_dir, args.shard_count, selection=args.selection)
    layouts = build_merge_layouts(source_shards, args.work_dir)
    total_bytes = sum(path.stat().st_size for path in source_shards)
    record_count = sum(len(_load_shard_records(path)) for path in source_shards)

    print(f"selection={args.selection} shards={[p.name for p in source_shards]}")
    print(f"records={record_count} total_bytes_gb={total_bytes / (1024**3):.3f}")
    print(f"warmup={args.warmup} repeats={args.repeats}")

    results: list[dict] = []
    labels = {
        1: f"{args.shard_count}x_small",
        2: "2x_merged",
        4: "4x_merged",
        8: "8x_merged",
    }
    for merge_factor in sorted(layouts):
        paths = layouts[merge_factor]
        if merge_factor == 1:
            bytes_total = total_bytes
        else:
            bytes_total = sum(path.stat().st_size for path in paths)
        results.append(
            summarize(
                labels[merge_factor],
                benchmark_paths(paths, repeats=args.repeats, warmup=args.warmup),
                file_count=len(paths),
                total_bytes=bytes_total,
            )
        )

    baseline = results[0]["median_s"]
    for row in results[1:]:
        speedup = baseline / row["median_s"]
        print(f"{row['label']} vs {results[0]['label']}: {speedup:.2f}x")

    summary_path = args.work_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
