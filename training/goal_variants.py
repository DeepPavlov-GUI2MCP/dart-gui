from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from uitars_format import build_user_prompt, UitarsFormatSettings
from uitars_trace_dataset import TraceScanSettings, repo_path

GOAL_VARIANT_PATTERN = re.compile(r"^goal_variant_(\d+)\.json$")
TIMESTAMP_RUN_PATTERN = re.compile(r"^\d{8}T\d{6}Z$")


@dataclass(frozen=True)
class GoalVariant:
    variant_index: int
    goal: str
    expected_outcome: str
    task_id: str
    source_path: Path
    task_index: int | None = None


def task_generation_path(value: str) -> Path:
    return repo_path(value)


def macro_state_id_from_task(task_payload: dict[str, Any], task_id: str) -> str:
    synthetic = task_payload.get("synthetic")
    if not isinstance(synthetic, dict):
        raise ValueError(f"Task {task_id} is missing synthetic metadata.")
    macro_state_id = synthetic.get("macro_state_id")
    if not isinstance(macro_state_id, str) or not macro_state_id.strip():
        raise ValueError(f"Task {task_id} is missing synthetic.macro_state_id.")
    return macro_state_id.strip()


def find_latest_generation_run(task_generation_root: Path, macro_state_id: str) -> Path:
    macro_dir = task_generation_root / macro_state_id
    if not macro_dir.is_dir():
        raise FileNotFoundError(f"Macro state directory not found: {macro_dir}")
    candidates: list[str] = []
    for child in macro_dir.iterdir():
        if not child.is_dir():
            continue
        if not TIMESTAMP_RUN_PATTERN.match(child.name):
            continue
        if (child / "goals").is_dir():
            candidates.append(child.name)
    if not candidates:
        raise FileNotFoundError(f"No timestamped goal generation runs under {macro_dir}")
    latest = max(candidates)
    return (macro_dir / latest).resolve()


def resolve_goals_task_dir(
    task_generation_root: Path,
    task_payload: dict[str, Any],
    task_id: str,
    *,
    run_cache: dict[str, Path] | None = None,
) -> Path:
    macro_state_id = macro_state_id_from_task(task_payload, task_id)
    if run_cache is not None and macro_state_id in run_cache:
        generation_run = run_cache[macro_state_id]
    else:
        generation_run = find_latest_generation_run(task_generation_root, macro_state_id)
        if run_cache is not None:
            run_cache[macro_state_id] = generation_run
    candidate = generation_run / "goals" / task_id
    if not candidate.is_dir():
        raise FileNotFoundError(f"Goal variants directory not found: {candidate}")
    return candidate.resolve()


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
        task_id=task_id.strip(),
        source_path=path.resolve(),
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


def load_variants_for_task(
    task_generation_root: Path,
    task_examples_dir: Path,
    domain: str,
    task_id: str,
    *,
    max_variants: int,
    run_cache: dict[str, Path] | None = None,
) -> list[GoalVariant]:
    task_payload = load_task_payload(task_examples_dir, domain, task_id)
    goals_task_dir = resolve_goals_task_dir(
        task_generation_root,
        task_payload,
        task_id,
        run_cache=run_cache,
    )
    variants = load_goal_variants(goals_task_dir, task_id, max_variants=max_variants)
    if not variants:
        raise FileNotFoundError(f"No goal variants found for task {task_id} in {goals_task_dir}")
    for variant in variants:
        if variant.task_index is not None:
            synthetic = task_payload.get("synthetic")
            if isinstance(synthetic, dict):
                task_index = synthetic.get("task_index")
                if isinstance(task_index, int) and task_index != variant.task_index:
                    raise ValueError(
                        f"Goal variant task_index {variant.task_index} does not match "
                        f"synthetic.task_index {task_index} for task {task_id}"
                    )
    return variants


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


def replace_instruction_in_messages(
    messages: Sequence[dict[str, Any]],
    instruction: str,
    settings: UitarsFormatSettings,
) -> list[dict[str, Any]]:
    if len(messages) < 2:
        raise ValueError("Expected at least system and user messages.")
    user_prompt = build_user_prompt(instruction, settings)
    updated: list[dict[str, Any]] = []
    for idx, message in enumerate(messages):
        if idx != 1 or message.get("role") != "user":
            updated.append(dict(message))
            continue
        content = message.get("content")
        if not isinstance(content, list):
            raise ValueError("Instruction user message must have list content.")
        new_content: list[Any] = []
        text_replaced = False
        for item in content:
            if (
                not text_replaced
                and isinstance(item, dict)
                and item.get("type") == "text"
            ):
                new_content.append({"type": "text", "text": user_prompt})
                text_replaced = True
            else:
                new_content.append(item)
        if not text_replaced:
            new_content.insert(0, {"type": "text", "text": user_prompt})
        updated.append({"role": "user", "content": new_content})
    return updated


def collect_task_ids_from_rows(rows: Sequence[dict[str, Any]]) -> set[str]:
    task_ids: set[str] = set()
    for row in rows:
        task_id = row.get("task_id")
        if isinstance(task_id, str) and task_id.strip():
            task_ids.add(task_id.strip())
    return task_ids


def infer_domain_from_rollout_dir(rollout_dir: str | Path) -> str:
    return Path(rollout_dir).parent.name


def assert_tasks_have_goal_variants(
    task_generation_root: Path,
    task_examples_dir: Path,
    rows: Sequence[dict[str, Any]],
    *,
    max_variants: int,
) -> dict[str, list[GoalVariant]]:
    task_ids = collect_task_ids_from_rows(rows)
    if not task_ids:
        raise ValueError("No task_id values found in staged rows.")
    domain_by_task: dict[str, str] = {}
    for row in rows:
        task_id = row.get("task_id")
        rollout_dir = row.get("rollout_dir")
        if not isinstance(task_id, str) or not isinstance(rollout_dir, str):
            continue
        domain_by_task.setdefault(task_id.strip(), infer_domain_from_rollout_dir(rollout_dir))
    missing: list[str] = []
    insufficient: list[str] = []
    catalog: dict[str, list[GoalVariant]] = {}
    run_cache: dict[str, Path] = {}
    for task_id in sorted(task_ids):
        domain = domain_by_task.get(task_id)
        if not domain:
            missing.append(task_id)
            continue
        try:
            variants = load_variants_for_task(
                task_generation_root,
                task_examples_dir,
                domain,
                task_id,
                max_variants=max_variants,
                run_cache=run_cache,
            )
        except (FileNotFoundError, ValueError):
            missing.append(task_id)
            continue
        if len(variants) < max_variants:
            insufficient.append(task_id)
        catalog[task_id] = variants
    errors: list[str] = []
    if missing:
        errors.append(f"tasks missing goal variants: {', '.join(missing)}")
    if insufficient:
        errors.append(
            f"tasks with fewer than {max_variants} goal variants: {', '.join(insufficient)}"
        )
    if errors:
        raise ValueError("; ".join(errors))
    return catalog


def task_examples_path(settings: TraceScanSettings) -> Path:
    return repo_path(settings.task_examples_dir)
