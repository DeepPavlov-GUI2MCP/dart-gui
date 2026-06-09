from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from goal_variants import infer_domain_from_rollout_dir, load_task_payload, task_generation_path
from uitars_trace_dataset import repo_path


SPLIT_FILENAME = "split.json"
SPLIT_FILENAME_SMOKE = "split-smoke.json"
SPLIT_STRATEGY = "one_holdout_task_per_microaction"
HOLDOUT_RULE = "min_task_index"
SMOKE_TASKS_PER_MICROACTION = 4
SMOKE_TRAIN_TASKS_PER_MICROACTION = 3


def normalize_pretokenized_split(split: str) -> str:
    if split == "dev":
        return "val"
    return split


@dataclass(frozen=True)
class TaskMicroactionMeta:
    task_id: str
    domain: str
    micro_action_id: str
    task_index: int


@dataclass(frozen=True)
class MicroactionSplit:
    strategy: str
    holdout_rule: str
    task_examples_dir: str
    task_generation_root: str | None
    train_task_ids: tuple[str, ...]
    val_task_ids: tuple[str, ...]
    microaction_holdouts: dict[str, str]
    train_rows: int
    val_rows: int

    def task_ids_for(self, split: str) -> set[str] | None:
        split = normalize_pretokenized_split(split)
        if split == "all":
            return None
        if split == "train":
            return set(self.train_task_ids)
        if split == "val":
            return set(self.val_task_ids)
        raise ValueError(f"Unknown pretokenized split: {split!r}")

    def row_count_for(self, split: str, *, total_rows: int | None = None) -> int:
        split = normalize_pretokenized_split(split)
        if split == "all":
            if total_rows is None:
                raise ValueError("total_rows is required when split='all'")
            return total_rows
        if split == "train":
            return self.train_rows
        if split == "val":
            return self.val_rows
        raise ValueError(f"Unknown pretokenized split: {split!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "holdout_rule": self.holdout_rule,
            "task_examples_dir": self.task_examples_dir,
            "task_generation_root": self.task_generation_root,
            "train_task_ids": list(self.train_task_ids),
            "val_task_ids": list(self.val_task_ids),
            "microaction_holdouts": dict(self.microaction_holdouts),
            "train_rows": self.train_rows,
            "val_rows": self.val_rows,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MicroactionSplit:
        val_task_ids = payload.get("val_task_ids")
        if val_task_ids is None:
            val_task_ids = payload.get("dev_task_ids", [])
        val_rows = payload.get("val_rows", payload.get("dev_rows", 0))
        return cls(
            strategy=str(payload.get("strategy") or SPLIT_STRATEGY),
            holdout_rule=str(payload.get("holdout_rule") or HOLDOUT_RULE),
            task_examples_dir=str(payload["task_examples_dir"]),
            task_generation_root=payload.get("task_generation_root"),
            train_task_ids=tuple(str(item) for item in payload.get("train_task_ids", [])),
            val_task_ids=tuple(str(item) for item in val_task_ids),
            microaction_holdouts={
                str(key): str(value)
                for key, value in (payload.get("microaction_holdouts") or {}).items()
            },
            train_rows=int(payload.get("train_rows", 0)),
            val_rows=int(val_rows),
        )


def split_filename(*, smoke_microactions: int | None = None) -> str:
    if smoke_microactions is not None:
        return SPLIT_FILENAME_SMOKE
    return SPLIT_FILENAME


def split_path(pretokenized_dir: Path, *, smoke_microactions: int | None = None) -> Path:
    return Path(pretokenized_dir) / split_filename(smoke_microactions=smoke_microactions)


def load_split(
    pretokenized_dir: Path,
    *,
    smoke_microactions: int | None = None,
) -> MicroactionSplit | None:
    path = split_path(pretokenized_dir, smoke_microactions=smoke_microactions)
    if not path.is_file() and smoke_microactions is not None:
        legacy = split_path(pretokenized_dir)
        if legacy.is_file():
            path = legacy
        else:
            return None
    elif not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return MicroactionSplit.from_dict(payload)


def write_split(
    pretokenized_dir: Path,
    split: MicroactionSplit,
    *,
    smoke_microactions: int | None = None,
) -> Path:
    path = split_path(pretokenized_dir, smoke_microactions=smoke_microactions)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def _resolve_osworld_task_path(
    task_payload: dict[str, Any],
    task_id: str,
    *,
    task_generation_root: Path | None,
) -> Path | None:
    synthetic = task_payload.get("synthetic")
    if not isinstance(synthetic, dict):
        return None
    generation_run = synthetic.get("generation_run")
    if isinstance(generation_run, str) and generation_run.strip():
        candidate = Path(generation_run.strip()) / "osworld" / f"{task_id}.json"
        if candidate.is_file():
            return candidate.resolve()
    if task_generation_root is None:
        return None
    macro_state_id = synthetic.get("macro_state_id")
    if not isinstance(macro_state_id, str) or not macro_state_id.strip():
        return None
    macro_dir = task_generation_root / macro_state_id.strip()
    if not macro_dir.is_dir():
        return None
    latest_run: Path | None = None
    for child in macro_dir.iterdir():
        if not child.is_dir():
            continue
        candidate = child / "osworld" / f"{task_id}.json"
        if candidate.is_file():
            if latest_run is None or child.name > latest_run.name:
                latest_run = child
    if latest_run is None:
        return None
    return (latest_run / "osworld" / f"{task_id}.json").resolve()


def _synthetic_fields(payload: dict[str, Any]) -> tuple[str | None, int | None]:
    synthetic = payload.get("synthetic")
    if not isinstance(synthetic, dict):
        return None, None
    micro_action_id = synthetic.get("micro_action_id")
    task_index = synthetic.get("task_index")
    micro = micro_action_id.strip() if isinstance(micro_action_id, str) and micro_action_id.strip() else None
    index = task_index if isinstance(task_index, int) else None
    return micro, index


def load_task_microaction_meta(
    *,
    task_examples_dir: Path,
    domain: str,
    task_id: str,
    task_generation_root: Path | None = None,
) -> TaskMicroactionMeta:
    task_payload = load_task_payload(task_examples_dir, domain, task_id)
    micro_action_id, task_index = _synthetic_fields(task_payload)
    if micro_action_id is None or task_index is None:
        osworld_path = _resolve_osworld_task_path(
            task_payload,
            task_id,
            task_generation_root=task_generation_root,
        )
        if osworld_path is not None:
            osworld_payload = json.loads(osworld_path.read_text(encoding="utf-8"))
            if isinstance(osworld_payload, dict):
                fallback_micro, fallback_index = _synthetic_fields(osworld_payload)
                micro_action_id = micro_action_id or fallback_micro
                task_index = task_index if task_index is not None else fallback_index
    if micro_action_id is None:
        raise ValueError(
            f"Task {task_id} is missing synthetic.micro_action_id in {task_examples_dir} "
            f"and gui_distillation osworld metadata."
        )
    if task_index is None:
        raise ValueError(
            f"Task {task_id} is missing synthetic.task_index in {task_examples_dir} "
            f"and gui_distillation osworld metadata."
        )
    return TaskMicroactionMeta(
        task_id=task_id,
        domain=domain,
        micro_action_id=micro_action_id,
        task_index=task_index,
    )


def load_metas_for_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    task_examples_dir: str,
    task_generation_root: str | None,
) -> list[TaskMicroactionMeta]:
    task_examples_path = repo_path(task_examples_dir)
    generation_root = task_generation_path(task_generation_root) if task_generation_root else None
    domains = domain_by_task_from_rows(rows)
    task_ids = sorted(domains)
    if not task_ids:
        raise ValueError("No task_id values found in staged rows.")
    return [
        load_task_microaction_meta(
            task_examples_dir=task_examples_path,
            domain=domains[task_id],
            task_id=task_id,
            task_generation_root=generation_root,
        )
        for task_id in task_ids
    ]


def group_by_microaction(metas: Sequence[TaskMicroactionMeta]) -> dict[str, list[TaskMicroactionMeta]]:
    by_microaction: dict[str, list[TaskMicroactionMeta]] = {}
    for meta in metas:
        by_microaction.setdefault(meta.micro_action_id, []).append(meta)
    return by_microaction


def select_smoke_microaction_groups(
    by_microaction: Mapping[str, Sequence[TaskMicroactionMeta]],
    *,
    num_microactions: int = 4,
    tasks_per_microaction: int = SMOKE_TASKS_PER_MICROACTION,
) -> dict[str, list[TaskMicroactionMeta]]:
    eligible = sorted(
        [
            (micro_action_id, list(group))
            for micro_action_id, group in by_microaction.items()
            if len(group) >= tasks_per_microaction
        ],
        key=lambda item: (-len(item[1]), item[0]),
    )
    if len(eligible) < num_microactions:
        raise ValueError(
            f"Need {num_microactions} microactions with at least {tasks_per_microaction} tasks each; "
            f"found {len(eligible)}."
        )
    selected: dict[str, list[TaskMicroactionMeta]] = {}
    for micro_action_id, group in eligible[:num_microactions]:
        ordered = sorted(group, key=lambda item: (item.task_index, item.task_id))
        selected[micro_action_id] = ordered[:tasks_per_microaction]
    return selected


def split_tasks_for_microaction_groups(
    groups: Mapping[str, Sequence[TaskMicroactionMeta]],
    *,
    rows_per_task: Mapping[str, int],
) -> tuple[list[str], list[str], dict[str, str], int, int]:
    val_task_ids: list[str] = []
    train_task_ids: list[str] = []
    holdouts: dict[str, str] = {}
    for micro_action_id, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda item: (item.task_index, item.task_id))
        holdout = ordered[0]
        holdouts[micro_action_id] = holdout.task_id
        val_task_ids.append(holdout.task_id)
        train_task_ids.extend(meta.task_id for meta in ordered[1:])
    train_rows = sum(rows_per_task.get(task_id, 0) for task_id in train_task_ids)
    val_rows = sum(rows_per_task.get(task_id, 0) for task_id in val_task_ids)
    return train_task_ids, val_task_ids, holdouts, train_rows, val_rows


def smoke_task_ids_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    task_examples_dir: str,
    task_generation_root: str | None,
    num_microactions: int = 4,
) -> set[str]:
    metas = load_metas_for_rows(
        rows,
        task_examples_dir=task_examples_dir,
        task_generation_root=task_generation_root,
    )
    by_microaction = group_by_microaction(metas)
    selected = select_smoke_microaction_groups(by_microaction, num_microactions=num_microactions)
    task_ids: set[str] = set()
    for group in selected.values():
        task_ids.update(meta.task_id for meta in group)
    return task_ids


def smoke_task_ids_from_rollouts(
    settings: Any,
    *,
    num_microactions: int = 4,
) -> set[str]:
    from uitars_trace_dataset import build_rows_for_rollout, infer_domain_and_task_id, iter_rollout_dirs

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for rollout_dir in iter_rollout_dirs(settings.trace_roots):
        try:
            _, task_id = infer_domain_and_task_id(rollout_dir)
        except (FileNotFoundError, ValueError):
            continue
        if task_id in seen:
            continue
        if not build_rows_for_rollout(rollout_dir, settings):
            continue
        seen.add(task_id)
        rows.append({"task_id": task_id, "rollout_dir": str(rollout_dir)})
    return smoke_task_ids_from_rows(
        rows,
        task_examples_dir=settings.task_examples_dir,
        task_generation_root=settings.task_generation_root,
        num_microactions=num_microactions,
    )


def domain_by_task_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in rows:
        task_id = row.get("task_id")
        rollout_dir = row.get("rollout_dir")
        if not isinstance(task_id, str) or not isinstance(rollout_dir, str):
            continue
        mapping.setdefault(task_id.strip(), infer_domain_from_rollout_dir(rollout_dir))
    return mapping


def compute_microaction_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    task_examples_dir: str,
    task_generation_root: str | None,
    rows_per_task: Mapping[str, int] | None = None,
) -> MicroactionSplit:
    task_examples_path = repo_path(task_examples_dir)
    generation_root = task_generation_path(task_generation_root) if task_generation_root else None
    metas = load_metas_for_rows(
        rows,
        task_examples_dir=task_examples_dir,
        task_generation_root=task_generation_root,
    )
    by_microaction = group_by_microaction(metas)
    groups = {
        micro_action_id: list(group)
        for micro_action_id, group in sorted(by_microaction.items())
    }
    task_ids = sorted({meta.task_id for meta in metas})
    if rows_per_task is None:
        rows_per_task = {task_id: 1 for task_id in task_ids}
    train_task_ids, val_task_ids, holdouts, train_rows, val_rows = split_tasks_for_microaction_groups(
        groups,
        rows_per_task=rows_per_task,
    )
    return MicroactionSplit(
        strategy=SPLIT_STRATEGY,
        holdout_rule=HOLDOUT_RULE,
        task_examples_dir=str(task_examples_path),
        task_generation_root=str(generation_root) if generation_root is not None else None,
        train_task_ids=tuple(train_task_ids),
        val_task_ids=tuple(sorted(val_task_ids)),
        microaction_holdouts=holdouts,
        train_rows=train_rows,
        val_rows=val_rows,
    )


def compute_smoke_microaction_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    task_examples_dir: str,
    task_generation_root: str | None,
    rows_per_task: Mapping[str, int] | None = None,
    num_microactions: int = 4,
) -> MicroactionSplit:
    task_examples_path = repo_path(task_examples_dir)
    generation_root = task_generation_path(task_generation_root) if task_generation_root else None
    metas = load_metas_for_rows(
        rows,
        task_examples_dir=task_examples_dir,
        task_generation_root=task_generation_root,
    )
    by_microaction = group_by_microaction(metas)
    groups = select_smoke_microaction_groups(by_microaction, num_microactions=num_microactions)
    included_task_ids = {meta.task_id for group in groups.values() for meta in group}
    if rows_per_task is None:
        rows_per_task = {meta.task_id: 1 for meta in metas if meta.task_id in included_task_ids}
    train_task_ids, val_task_ids, holdouts, train_rows, val_rows = split_tasks_for_microaction_groups(
        groups,
        rows_per_task=rows_per_task,
    )
    return MicroactionSplit(
        strategy=SPLIT_STRATEGY,
        holdout_rule=HOLDOUT_RULE,
        task_examples_dir=str(task_examples_path),
        task_generation_root=str(generation_root) if generation_root is not None else None,
        train_task_ids=tuple(train_task_ids),
        val_task_ids=tuple(sorted(val_task_ids)),
        microaction_holdouts=holdouts,
        train_rows=train_rows,
        val_rows=val_rows,
    )


def compute_split_for_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    task_examples_dir: str,
    task_generation_root: str | None,
    rows_per_task: Mapping[str, int] | None = None,
    smoke_microactions: int | None = None,
) -> MicroactionSplit:
    if smoke_microactions is not None:
        return compute_smoke_microaction_split(
            rows,
            task_examples_dir=task_examples_dir,
            task_generation_root=task_generation_root,
            rows_per_task=rows_per_task,
            num_microactions=smoke_microactions,
        )
    return compute_microaction_split(
        rows,
        task_examples_dir=task_examples_dir,
        task_generation_root=task_generation_root,
        rows_per_task=rows_per_task,
    )


def compute_split_for_training_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    task_examples_dir: str,
    task_generation_root: str | None,
    goal_variants: bool = False,
    max_goal_variants: int = 1,
    smoke_microactions: int | None = None,
) -> MicroactionSplit:
    rows_per_task = rows_per_task_from_catalog(rows, goal_variants=False, catalog=None)
    if goal_variants:
        if not task_generation_root:
            raise ValueError("task_generation_root is required when goal_variants is enabled.")
        from goal_variants import assert_tasks_have_goal_variants, task_generation_path

        catalog = assert_tasks_have_goal_variants(
            task_generation_path(task_generation_root),
            repo_path(task_examples_dir),
            rows,
            max_variants=max_goal_variants,
        )
        rows_per_task = rows_per_task_from_catalog(
            rows,
            goal_variants=True,
            catalog=catalog,
        )
    return compute_split_for_rows(
        rows,
        task_examples_dir=task_examples_dir,
        task_generation_root=task_generation_root,
        rows_per_task=rows_per_task,
        smoke_microactions=smoke_microactions,
    )


def validate_split_for_training_rows(
    split: MicroactionSplit,
    rows: Sequence[Mapping[str, Any]],
    *,
    task_examples_dir: str,
    task_generation_root: str | None,
    goal_variants: bool = False,
    max_goal_variants: int = 1,
    smoke_microactions: int | None = None,
) -> None:
    expected = compute_split_for_training_rows(
        rows,
        task_examples_dir=task_examples_dir,
        task_generation_root=task_generation_root,
        goal_variants=goal_variants,
        max_goal_variants=max_goal_variants,
        smoke_microactions=smoke_microactions,
    )
    if expected.to_dict() != split.to_dict():
        raise ValueError(
            "pretokenized split.json does not match the microaction holdout rule "
            f"(expected strategy={expected.strategy!r}, found strategy={split.strategy!r}; "
            f"expected val tasks={len(expected.val_task_ids)}, found={len(split.val_task_ids)})"
        )


def rows_per_task_from_catalog(
    rows: Sequence[Mapping[str, Any]],
    *,
    goal_variants: bool,
    catalog: Mapping[str, Sequence[Any]] | None,
) -> dict[str, int]:
    step_counts: dict[str, int] = {}
    for row in rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str):
            continue
        key = task_id.strip()
        step_counts[key] = step_counts.get(key, 0) + 1
    counts: dict[str, int] = {}
    for task_id, steps in step_counts.items():
        multiplier = 1
        if goal_variants and catalog is not None:
            variants = catalog.get(task_id)
            multiplier = len(variants) if variants else 0
        counts[task_id] = steps * multiplier
    return counts
