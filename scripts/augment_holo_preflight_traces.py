#!/usr/bin/env python3
"""Enrich Holo trace preflight steps with Holotron-style reasoning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "training"))
sys.path.insert(0, str(REPO_ROOT / "GUI-Docker-Env"))

from preflight_trace_augmentation import (  # noqa: E402
    AugmentSettings,
    resolve_task_examples_dir,
    resolve_trace_root,
    run_augmentation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-root",
        type=Path,
        required=True,
        help="Holo results directory containing rollout traces.",
    )
    parser.add_argument(
        "--task-examples-dir",
        type=Path,
        default=Path("GUI-Docker-Env/evaluation_examples/examples"),
        help="Directory with OSWorld task JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for per-task augmented JSON files (default: trace-root/preflight_augmented/<mode>).",
    )
    parser.add_argument(
        "--mode",
        choices=["openai-api", "holo-native"],
        required=True,
        help="LLM backend for preflight enrichment.",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=1,
        help="Number of tasks to process after --skip-first-n.",
    )
    parser.add_argument(
        "--skip-first-n",
        type=int,
        default=0,
        help="Skip the first N tasks from traces.jsonl before processing.",
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=[],
        help="Process only these task IDs (ignores --n-tasks and --skip-first-n).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate outputs and refresh the mode cache.",
    )
    parser.add_argument(
        "--write-traces-manifest",
        action="store_true",
        help="Write discovered rollouts to traces.jsonl under trace-root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve tasks and cache hits without calling LLMs or writing outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = None
    if args.output_dir is not None:
        output_dir = resolve_trace_root(args.output_dir)
    settings = AugmentSettings(
        trace_root=resolve_trace_root(args.trace_root),
        task_examples_dir=resolve_task_examples_dir(args.task_examples_dir),
        mode=args.mode,
        n_tasks=args.n_tasks,
        skip_first_n=args.skip_first_n,
        task_ids=tuple(args.task_ids),
        output_dir=output_dir,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        write_traces_manifest=args.write_traces_manifest,
    )
    summary = run_augmentation(settings)
    print(json.dumps(summary, indent=2))
    return 1 if summary.get("failures") else 0


if __name__ == "__main__":
    raise SystemExit(main())
