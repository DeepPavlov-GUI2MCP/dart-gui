#!/usr/bin/env python3
"""Write goal-variant OSWorld task JSONs for existing Holo rollout traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "training"))

from trace_goal_replacement import (  # noqa: E402
    GoalReplaceSettings,
    resolve_output_dir,
    resolve_task_examples_dir,
    resolve_trace_root,
    run_goal_replacement,
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
        default=Path("GUI-Docker-Env/evaluation_examples/examples_goal_variants"),
        help="Root directory for per-variant augmented task JSON trees.",
    )
    parser.add_argument(
        "--goals-root",
        type=Path,
        default=None,
        help="Optional root to search for goals/{task_id} when synthetic.generation_run is missing.",
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=1,
        help="Maximum goal variants to emit per trace (default: 1).",
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=[],
        help="Process only these task IDs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing augmented task JSON files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve tasks and report without writing outputs.",
    )
    parser.add_argument(
        "--require-all-goals",
        action="store_true",
        help="Fail when any trace has fewer than --max-variants goal files available.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_variants < 1:
        print("error: --max-variants must be at least 1", file=sys.stderr)
        return 2
    goals_root = None
    if args.goals_root is not None:
        goals_root = resolve_trace_root(args.goals_root)
    settings = GoalReplaceSettings(
        trace_root=resolve_trace_root(args.trace_root),
        task_examples_dir=resolve_task_examples_dir(args.task_examples_dir),
        output_dir=resolve_output_dir(args.output_dir),
        max_variants=args.max_variants,
        goals_root=goals_root,
        task_ids=tuple(args.task_ids),
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        require_all_goals=args.require_all_goals,
    )
    summary = run_goal_replacement(settings)
    print(json.dumps(summary, indent=2))
    if summary.get("failures") or summary.get("require_all_goals_failed"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
