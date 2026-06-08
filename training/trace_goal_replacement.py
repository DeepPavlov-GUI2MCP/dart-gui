from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from uitars_trace_dataset import infer_domain_and_task_id, iter_rollout_dirs, repo_path

REPO_ROOT = Path(__file__).resolve().parents[1]
GOAL_VARIANT_PATTERN = re.compile(r"^goal_variant_(\d+)\.json$")
MANIFEST_FILENAME = "goal_replacement_manifest.jsonl"


@dataclass(frozen=True)
class GoalVariant:
    variant_index: int
    goal: str
    expected_outcome: str
    source_path: Path
    task_id: str
    task_index: int | None = None


@dataclass(frozen=True)
class GoalReplaceSettings:
    trace_root: Path
    task_examples_dir: Path
    output_dir: Path
    max_variants: int = 1
    goals_root: Path | None = None
    task_ids: tuple[str, ...] = ()
    overwrite: bool = False
    dry_run: bool = False
    require_all_goals: bool = False


@dataclass
class TaskReplaceResult:
    task_id: str
    status: str
    variants_written: int = 0
    error: str | None = None


def resolve_trace_root(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def resolve_task_examples_dir(value: str | Path) -> Path:
    return repo_path(str(value))


def resolve_output_dir(value: str | Path) -> Path:
    return repo_path(str(value))


def load_task_payload(task_examples_dir: Path, domain: str, task_id: str) -> dict[str, Any]:
    task_path = task_examples_dir / domain / f"{task_id}.json"
    if not task_path.is_file():
        raise FileNotFoundError(f"Task example not found: {task_path}")
    with open(task_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Task example must be a JSON object: {task_path}")
    task_json_id = payload.get("id")
    if task_json_id != task_id:
        raise ValueError(
            f"Task id mismatch for {task_path}: rollout={task_id!r} json={task_json_id!r}"
        )
    return payload


def resolve_goals_task_dir(
    task_payload: dict[str, Any],
    task_id: str,
    *,
    goals_root: Path | None,
) -> Path | None:
    synthetic = task_payload.get("synthetic")
    if isinstance(synthetic, dict):
        generation_run = synthetic.get("generation_run")
        if isinstance(generation_run, str) and generation_run.strip():
            candidate = Path(generation_run.strip()).expanduser() / "goals" / task_id
            if candidate.is_dir():
                return candidate.resolve()
    if goals_root is not None:
        matches = sorted(goals_root.glob(f"**/goals/{task_id}"))
        for match in matches:
            if match.is_dir():
                return match.resolve()
    return None


def parse_goal_variant(path: Path) -> GoalVariant | None:
    match = GOAL_VARIANT_PATTERN.match(path.name)
    if not match:
        return None
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return None
    goal = payload.get("goal")
    expected_outcome = payload.get("expected_outcome")
    task_id = payload.get("id")
    if not isinstance(goal, str) or not goal.strip():
        return None
    if not isinstance(expected_outcome, str) or not expected_outcome.strip():
        return None
    if not isinstance(task_id, str) or not task_id.strip():
        return None
    variant_index = payload.get("variant_index")
    if not isinstance(variant_index, int):
        variant_index = int(match.group(1))
    task_index = payload.get("task_index")
    if task_index is not None and not isinstance(task_index, int):
        task_index = None
    return GoalVariant(
        variant_index=variant_index,
        goal=goal.strip(),
        expected_outcome=expected_outcome.strip(),
        source_path=path.resolve(),
        task_id=task_id.strip(),
        task_index=task_index,
    )


def load_goal_variants(goals_task_dir: Path, task_id: str, *, max_variants: int) -> list[GoalVariant]:
    if max_variants < 1:
        raise ValueError("max_variants must be at least 1")
    variants: list[GoalVariant] = []
    for path in sorted(goals_task_dir.glob("goal_variant_*.json")):
        parsed = parse_goal_variant(path)
        if parsed is None:
            continue
        if parsed.task_id != task_id:
            raise ValueError(
                f"Goal variant task id mismatch in {path}: "
                f"expected={task_id!r} got={parsed.task_id!r}"
            )
        variants.append(parsed)
    variants.sort(key=lambda item: item.variant_index)
    return variants[:max_variants]


def validate_variant_task_index(task_payload: dict[str, Any], variant: GoalVariant) -> None:
    if variant.task_index is None:
        return
    synthetic = task_payload.get("synthetic")
    if not isinstance(synthetic, dict):
        return
    task_index = synthetic.get("task_index")
    if isinstance(task_index, int) and task_index != variant.task_index:
        raise ValueError(
            f"Goal variant task_index {variant.task_index} does not match "
            f"synthetic.task_index {task_index} for task {variant.task_id}"
        )


def build_augmented_task(base_task: dict[str, Any], variant: GoalVariant) -> dict[str, Any]:
    augmented = copy.deepcopy(base_task)
    original_instruction = augmented.get("instruction")
    if not isinstance(original_instruction, str):
        original_instruction = ""
    augmented["instruction"] = variant.goal
    synthetic = augmented.get("synthetic")
    if not isinstance(synthetic, dict):
        synthetic = {}
        augmented["synthetic"] = synthetic
    synthetic["goal_variant_index"] = variant.variant_index
    synthetic["original_instruction"] = original_instruction.strip()
    synthetic["goal_variant_expected_outcome"] = variant.expected_outcome
    synthetic["goal_variant_source"] = str(variant.source_path)
    return augmented


def augmented_task_output_path(
    output_dir: Path,
    *,
    variant_index: int,
    domain: str,
    task_id: str,
) -> Path:
    return output_dir / f"variant_{variant_index}" / domain / f"{task_id}.json"


def repo_relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def select_rollout_dirs(trace_root: Path, task_ids: tuple[str, ...]) -> list[Path]:
    rollout_dirs = list(iter_rollout_dirs((str(trace_root),)))
    if task_ids:
        allowed = set(task_ids)
        rollout_dirs = [path for path in rollout_dirs if path.name in allowed]
    return sorted(rollout_dirs, key=lambda path: path.name)


def process_rollout(
    rollout_dir: Path,
    settings: GoalReplaceSettings,
    *,
    manifest_rows: list[dict[str, Any]],
) -> TaskReplaceResult:
    domain, task_id = infer_domain_and_task_id(rollout_dir)
    try:
        task_payload = load_task_payload(settings.task_examples_dir, domain, task_id)
        goals_task_dir = resolve_goals_task_dir(
            task_payload,
            task_id,
            goals_root=settings.goals_root,
        )
        if goals_task_dir is None:
            if settings.require_all_goals:
                return TaskReplaceResult(
                    task_id=task_id,
                    status="failed",
                    error=f"no goals directory found for task {task_id}",
                )
            return TaskReplaceResult(task_id=task_id, status="skipped_no_goals")
        variants = load_goal_variants(goals_task_dir, task_id, max_variants=settings.max_variants)
        if not variants:
            if settings.require_all_goals:
                return TaskReplaceResult(
                    task_id=task_id,
                    status="failed",
                    error=f"no goal variants found in {goals_task_dir}",
                )
            return TaskReplaceResult(task_id=task_id, status="skipped_no_goals")
        if len(variants) < settings.max_variants and settings.require_all_goals:
            return TaskReplaceResult(
                task_id=task_id,
                status="failed",
                error=(
                    f"found {len(variants)} goal variant(s) in {goals_task_dir}, "
                    f"expected {settings.max_variants}"
                ),
            )
        written = 0
        rollout_rel = rollout_dir.relative_to(settings.trace_root).as_posix()
        for variant in variants:
            validate_variant_task_index(task_payload, variant)
            output_path = augmented_task_output_path(
                settings.output_dir,
                variant_index=variant.variant_index,
                domain=domain,
                task_id=task_id,
            )
            if output_path.is_file() and not settings.overwrite:
                manifest_rows.append(
                    {
                        "task_id": task_id,
                        "domain": domain,
                        "rollout_dir": rollout_rel,
                        "variant_index": variant.variant_index,
                        "instruction": variant.goal,
                        "original_instruction": str(task_payload.get("instruction") or "").strip(),
                        "augmented_task_path": repo_relative_path(output_path),
                        "goal_source_path": str(variant.source_path),
                        "status": "skipped_exists",
                    }
                )
                continue
            if not settings.dry_run:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                augmented = build_augmented_task(task_payload, variant)
                output_path.write_text(
                    json.dumps(augmented, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            manifest_rows.append(
                {
                    "task_id": task_id,
                    "domain": domain,
                    "rollout_dir": rollout_rel,
                    "variant_index": variant.variant_index,
                    "instruction": variant.goal,
                    "original_instruction": str(task_payload.get("instruction") or "").strip(),
                    "augmented_task_path": repo_relative_path(output_path),
                    "goal_source_path": str(variant.source_path),
                    "status": "dry_run" if settings.dry_run else "written",
                }
            )
            written += 1
        return TaskReplaceResult(task_id=task_id, status="ok", variants_written=written)
    except (FileNotFoundError, ValueError) as exc:
        return TaskReplaceResult(task_id=task_id, status="failed", error=str(exc))


def write_manifest(trace_root: Path, rows: list[dict[str, Any]], *, dry_run: bool) -> Path:
    manifest_path = trace_root / MANIFEST_FILENAME
    if dry_run:
        return manifest_path
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest_path


def run_goal_replacement(settings: GoalReplaceSettings) -> dict[str, Any]:
    if settings.max_variants < 1:
        raise ValueError("max_variants must be at least 1")
    rollout_dirs = select_rollout_dirs(settings.trace_root, settings.task_ids)
    manifest_rows: list[dict[str, Any]] = []
    results: list[TaskReplaceResult] = []
    for rollout_dir in rollout_dirs:
        results.append(
            process_rollout(rollout_dir, settings, manifest_rows=manifest_rows)
        )

    summary = {
        "trace_root": str(settings.trace_root),
        "task_examples_dir": str(settings.task_examples_dir),
        "output_dir": str(settings.output_dir),
        "max_variants": settings.max_variants,
        "dry_run": settings.dry_run,
        "rollouts_discovered": len(rollout_dirs),
        "processed_ok": sum(1 for item in results if item.status == "ok"),
        "skipped_no_goals": sum(1 for item in results if item.status == "skipped_no_goals"),
        "variants_written": sum(item.variants_written for item in results),
        "failures": [
            {"task_id": item.task_id, "error": item.error}
            for item in results
            if item.status == "failed"
        ],
    }
    if manifest_rows and not settings.dry_run:
        manifest_path = write_manifest(settings.trace_root, manifest_rows, dry_run=False)
        summary["manifest_path"] = str(manifest_path)
    elif manifest_rows:
        summary["manifest_path"] = str(settings.trace_root / MANIFEST_FILENAME)
        summary["manifest_rows"] = len(manifest_rows)
    if settings.require_all_goals and summary["failures"]:
        summary["require_all_goals_failed"] = True
    return summary
