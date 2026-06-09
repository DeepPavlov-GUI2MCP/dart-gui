#!/usr/bin/env python3
"""Validate pretokenized SFT dataset before ML Space submission."""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TRAINING_DIR = _REPO_ROOT / "training"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

from goal_variants import assert_tasks_have_goal_variants, task_examples_path, task_generation_path
from jobs.hf_cache import apply_hf_hub_env, warn_missing_hf_token
from uitars_trace_dataset import TraceScanSettings
from pretokenized_manifest import (
    collect_pretokenized_stats,
    collect_pretokenized_stats_fast,
    count_jsonl_rows,
    expected_training_rows_for_goal_variants,
    load_jsonl_rows,
    load_manifest,
    manifest_supports_fast_validation,
    validate_manifest_metadata,
)
import run_sft

LOGGER = logging.getLogger(__name__)


def repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


@contextmanager
def stage(name: str):
    started = time.perf_counter()
    LOGGER.info("[presubmit] %s started", name)
    try:
        yield
    finally:
        LOGGER.info("[presubmit] %s done in %.3fs", name, time.perf_counter() - started)


def resolve_pretokenized_dir(config: run_sft.ResolvedConfig) -> Path:
    override = os.environ.get("DART_SFT_PRETOKENIZED_DIR", "").strip()
    if override:
        return repo_path(override)
    if config.dataset.pretokenized_traces_dir:
        return repo_path(config.dataset.pretokenized_traces_dir)
    fail("dataset.pretokenized_traces_dir must be set for the SFT job")


def validate_pretokenized_dataset(config_path: str) -> run_sft.ResolvedConfig:
    with stage("load config"):
        config = run_sft.load_config_file(config_path)
        if config.dataset.format != "uitars_trace":
            fail("dataset.format must be uitars_trace")
        if not config.dataset.pretokenized_traces_dir and not os.environ.get("DART_SFT_PRETOKENIZED_DIR", "").strip():
            fail("dataset.pretokenized_traces_dir must be set for the SFT job")

    with stage("resolve paths"):
        staged_jsonl = run_sft.resolve_staged_data_path(config)
        if not staged_jsonl.is_file():
            fail(f"staged JSONL not found: {staged_jsonl}")
        pretokenized_dir = resolve_pretokenized_dir(config)
        if not pretokenized_dir.is_dir():
            fail(f"pretokenized directory not found: {pretokenized_dir}")

    rows: list[dict] = []
    with stage("load staged rows"):
        if config.dataset.goal_variants:
            rows = load_jsonl_rows(staged_jsonl)
            expected_base_rows = len(rows)
        else:
            expected_base_rows = count_jsonl_rows(staged_jsonl)
        if expected_base_rows == 0:
            fail(f"staged JSONL is empty: {staged_jsonl}")

    with stage("compute expected training rows"):
        if config.dataset.goal_variants:
            if not config.dataset.task_generation_root:
                fail("dataset.task_generation_root is required when goal_variants is enabled")
            catalog = assert_tasks_have_goal_variants(
                task_generation_path(config.dataset.task_generation_root),
                task_examples_path(
                    TraceScanSettings(
                        trace_roots=config.dataset.trace_roots,
                        task_examples_dir=config.dataset.task_examples_dir or "",
                        sample_mode=config.dataset.sample_mode,
                        history_n=config.dataset.history_n,
                        min_result=config.dataset.min_result,
                        trace_source=config.dataset.trace_source,  # type: ignore[arg-type]
                        uitars=config.dataset.uitars,
                        preflight_augmented_path=config.dataset.preflight_augmented_path,
                        trace_root=config.dataset.trace_root,
                        max_rollouts=config.dataset.max_rollouts,
                        max_preflight_steps=config.dataset.max_preflight_steps,
                        goal_variants=config.dataset.goal_variants,
                        max_goal_variants=config.dataset.max_goal_variants,
                        task_generation_root=config.dataset.task_generation_root,
                        pretokenized_traces_dir=config.dataset.pretokenized_traces_dir,
                    )
                ),
                rows,
                max_variants=config.dataset.max_goal_variants,
            )
            expected_full_rows = expected_training_rows_for_goal_variants(rows, catalog)
        else:
            expected_full_rows = expected_base_rows

    with stage("load manifest"):
        manifest = load_manifest(pretokenized_dir)

    with stage("validate pretokenized shards"):
        try:
            if manifest is not None and manifest_supports_fast_validation(manifest):
                LOGGER.info("[presubmit] using fast manifest validation")
                stats = collect_pretokenized_stats_fast(pretokenized_dir, manifest)
            else:
                LOGGER.info("[presubmit] manifest absent; falling back to full shard scan")
                stats = collect_pretokenized_stats(pretokenized_dir)
        except ValueError as exc:
            fail(str(exc))

    with stage("validate split metadata"):
        from microaction_split import load_split, validate_split_for_training_rows

        split = load_split(pretokenized_dir)
        if split is None:
            fail(f"missing pretokenized split metadata: {pretokenized_dir / 'split.json'}")
        split_rows = rows if rows else load_jsonl_rows(staged_jsonl)
        try:
            validate_split_for_training_rows(
                split,
                split_rows,
                task_examples_dir=config.dataset.task_examples_dir or "",
                task_generation_root=config.dataset.task_generation_root,
                goal_variants=config.dataset.goal_variants,
                max_goal_variants=config.dataset.max_goal_variants,
                smoke_microactions=config.dataset.smoke_microactions,
            )
        except ValueError as exc:
            fail(str(exc))
        if split.train_rows + split.val_rows != expected_full_rows:
            fail(
                "pretokenized split row count mismatch: "
                f"train={split.train_rows} val={split.val_rows} "
                f"expected total={expected_full_rows}"
            )
        pretokenized_split = config.dataset.pretokenized_split
        if pretokenized_split == "dev":
            pretokenized_split = "val"
        if pretokenized_split == "train":
            split_rows = split.train_rows
        elif pretokenized_split == "val":
            split_rows = split.val_rows
        else:
            split_rows = expected_full_rows

    with stage("compare row counts"):
        if stats.actual_rows != expected_full_rows:
            fail(
                "pretokenized row count mismatch: "
                f"found {stats.actual_rows}, expected {expected_full_rows}"
            )

    with stage("validate manifest metadata"):
        if manifest is not None:
            manifest_errors = validate_manifest_metadata(
                manifest,
                goal_variants=config.dataset.goal_variants,
                max_goal_variants=config.dataset.max_goal_variants,
                task_generation_root=config.dataset.task_generation_root,
                base_rows=expected_base_rows,
                expanded_rows=expected_full_rows,
                rollouts=stats.rollouts,
                tasks=stats.tasks,
                shard_count=stats.shard_count,
                base_model=config.model.base_model,
            )
            if manifest_errors:
                fail("; ".join(manifest_errors))

    goals_label = str(stats.goals) if stats.goals is not None else "N/A"
    manifest_label = "present" if manifest is not None else "absent"
    print(f"Pretokenized dataset: {pretokenized_dir}")
    print(f"Staged JSONL: {staged_jsonl}")
    print(f"Rollouts: {stats.rollouts}")
    print(f"Tasks: {stats.tasks}")
    print(f"Goals: {goals_label}")
    print(f"Training rows: {stats.actual_rows} (expected {expected_full_rows})")
    print(f"Split train/val rows: {split.train_rows}/{split.val_rows} (config split={config.dataset.pretokenized_split}, rows={split_rows})")
    print(f"Shards: {stats.shard_count}  Manifest: {manifest_label}")
    return config


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    apply_hf_hub_env()
    warn_missing_hf_token()
    config_path = os.environ.get("DART_SFT_CONFIG", "").strip()
    if not config_path:
        fail("DART_SFT_CONFIG must be set")
    started = time.perf_counter()
    validate_pretokenized_dataset(config_path)
    LOGGER.info("[presubmit] total wall time %.3fs", time.perf_counter() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
