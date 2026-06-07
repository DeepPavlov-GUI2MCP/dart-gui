from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

from holo_to_uitars import convert_holo_response, parse_holo_response
from uitars_format import UitarsFormatSettings, build_messages_from_images_and_responses


REPO_ROOT = Path(__file__).resolve().parents[1]
TraceSource = Literal["auto", "uitars", "holo"]


@dataclass(frozen=True)
class TraceStep:
    step_num: int
    response: str
    screenshot_file: str


@dataclass(frozen=True)
class TrajRow:
    step_num: int
    screenshot_file: str
    response: str | None
    action: Any


@dataclass(frozen=True)
class TraceScanSettings:
    trace_roots: tuple[str, ...]
    task_examples_dir: str
    sample_mode: str = "per_step"
    history_n: int = 5
    min_result: float | None = None
    trace_source: TraceSource = "auto"
    uitars: UitarsFormatSettings = UitarsFormatSettings()


def repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_traj_rows(rollout_dir: Path) -> list[TrajRow]:
    traj_path = rollout_dir / "traj.jsonl"
    if not traj_path.is_file():
        return []
    rows: list[TrajRow] = []
    with open(traj_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            step_num = payload.get("step_num")
            screenshot_file = payload.get("screenshot_file")
            if not isinstance(step_num, int):
                continue
            if not isinstance(screenshot_file, str) or not screenshot_file.strip():
                continue
            response = payload.get("response")
            if response is not None and not isinstance(response, str):
                continue
            rows.append(
                TrajRow(
                    step_num=step_num,
                    screenshot_file=screenshot_file.strip(),
                    response=response.strip() if isinstance(response, str) and response.strip() else None,
                    action=payload.get("action"),
                )
            )
    rows.sort(key=lambda row: row.step_num)
    return rows


def is_preflight_row(row: TrajRow) -> bool:
    action = row.action
    return isinstance(action, dict) and action.get("phase") == "a11y_preflight"


def is_holo_agent_row(row: TrajRow) -> bool:
    if not row.response or is_preflight_row(row):
        return False
    try:
        parse_holo_response(row.response)
    except Exception:
        return False
    return True


def detect_rollout_trace_source(rows: list[TrajRow], configured: TraceSource) -> str:
    if configured in {"uitars", "holo"}:
        return configured
    if any(is_preflight_row(row) for row in rows):
        return "holo"
    for row in rows:
        if row.response and is_holo_agent_row(row):
            return "holo"
    return "uitars"


def holo_agent_row_indices(rows: list[TrajRow]) -> list[int]:
    return [idx for idx, row in enumerate(rows) if is_holo_agent_row(row)]


def load_traj_steps(rollout_dir: Path) -> list[TraceStep]:
    steps: list[TraceStep] = []
    for row in load_traj_rows(rollout_dir):
        if not row.response:
            continue
        steps.append(
            TraceStep(
                step_num=row.step_num,
                response=row.response,
                screenshot_file=row.screenshot_file,
            )
        )
    return steps


def observation_paths_for_steps(rollout_dir: Path, steps: list[TraceStep]) -> list[Path]:
    return [rollout_dir / step.screenshot_file for step in steps]


def holo_observation_paths(
    rollout_dir: Path,
    rows: list[TrajRow],
    agent_indices: list[int],
    agent_step_idx: int,
) -> list[Path]:
    first_agent_idx = agent_indices[0]
    paths: list[Path] = []
    for offset in range(agent_step_idx + 1):
        row_idx = first_agent_idx - 1 + offset
        if row_idx < 0:
            raise ValueError("Holo trace is missing a pre-action screenshot before the first agent step.")
        paths.append(rollout_dir / rows[row_idx].screenshot_file)
    return paths


def holo_converted_responses(rows: list[TrajRow], agent_indices: list[int]) -> list[str]:
    return [convert_holo_response(rows[idx].response or "") for idx in agent_indices]


def read_result_score(rollout_dir: Path) -> float | None:
    result_path = rollout_dir / "result.txt"
    if not result_path.is_file():
        return None
    text = result_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    try:
        return float(text.splitlines()[0])
    except ValueError:
        return None


def infer_domain_and_task_id(rollout_dir: Path) -> tuple[str, str]:
    task_id = rollout_dir.name
    domain = rollout_dir.parent.name
    if not task_id or not domain:
        raise ValueError(f"Could not infer domain/task id from rollout dir: {rollout_dir}")
    return domain, task_id


def load_task_instruction(task_examples_dir: Path, domain: str, task_id: str) -> str:
    task_path = task_examples_dir / domain / f"{task_id}.json"
    if not task_path.is_file():
        raise FileNotFoundError(f"Task example not found: {task_path}")
    with open(task_path, encoding="utf-8") as f:
        payload = json.load(f)
    instruction = payload.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"Missing instruction in task example: {task_path}")
    return instruction.strip()


def iter_rollout_dirs(trace_roots: Sequence[str]) -> Iterator[Path]:
    seen: set[Path] = set()
    for root_value in trace_roots:
        root = repo_path(root_value)
        if not root.exists():
            continue
        for traj_path in root.rglob("traj.jsonl"):
            rollout_dir = traj_path.parent.resolve()
            if rollout_dir in seen:
                continue
            seen.add(rollout_dir)
            yield rollout_dir


def _format_settings(settings: TraceScanSettings) -> UitarsFormatSettings:
    return UitarsFormatSettings(
        prompt_style=settings.uitars.prompt_style,
        infer_mode=settings.uitars.infer_mode,
        language=settings.uitars.language,
        max_pixels=settings.uitars.max_pixels,
        min_pixels=settings.uitars.min_pixels,
        history_n=settings.history_n,
    )


def build_uitars_per_step_rows(
    rollout_dir: Path,
    instruction: str,
    settings: TraceScanSettings,
) -> list[dict[str, Any]]:
    steps = load_traj_steps(rollout_dir)
    if not steps:
        return []
    obs_paths = observation_paths_for_steps(rollout_dir, steps)
    format_settings = _format_settings(settings)
    rows: list[dict[str, Any]] = []
    for step_idx, step in enumerate(steps):
        image_paths = obs_paths[: step_idx + 1]
        if not all(path.is_file() for path in image_paths):
            continue
        history_responses = [item.response for item in steps[:step_idx]]
        messages = build_messages_from_images_and_responses(
            instruction,
            image_paths,
            history_responses,
            format_settings,
            target_response=step.response,
            stage_relative_paths=True,
            rollout_dir=rollout_dir,
        )
        rows.append(
            {
                "messages": messages,
                "task_id": infer_domain_and_task_id(rollout_dir)[1],
                "rollout_dir": str(rollout_dir),
                "step_num": step.step_num,
                "sample_mode": "per_step",
                "loss_on_last_assistant_only": True,
            }
        )
    return rows


def build_holo_per_step_rows(
    rollout_dir: Path,
    instruction: str,
    settings: TraceScanSettings,
    rows: list[TrajRow],
) -> list[dict[str, Any]]:
    agent_indices = holo_agent_row_indices(rows)
    if not agent_indices:
        return []
    converted = holo_converted_responses(rows, agent_indices)
    format_settings = _format_settings(settings)
    built: list[dict[str, Any]] = []
    for agent_step_idx, traj_idx in enumerate(agent_indices):
        image_paths = holo_observation_paths(rollout_dir, rows, agent_indices, agent_step_idx)
        if not all(path.is_file() for path in image_paths):
            continue
        messages = build_messages_from_images_and_responses(
            instruction,
            image_paths,
            converted[:agent_step_idx],
            format_settings,
            target_response=converted[agent_step_idx],
            stage_relative_paths=True,
            rollout_dir=rollout_dir,
        )
        built.append(
            {
                "messages": messages,
                "task_id": infer_domain_and_task_id(rollout_dir)[1],
                "rollout_dir": str(rollout_dir),
                "step_num": rows[traj_idx].step_num,
                "sample_mode": "per_step",
                "loss_on_last_assistant_only": True,
            }
        )
    return built


def build_uitars_full_trajectory_row(
    rollout_dir: Path,
    instruction: str,
    settings: TraceScanSettings,
) -> dict[str, Any] | None:
    steps = load_traj_steps(rollout_dir)
    if not steps:
        return None
    obs_paths = observation_paths_for_steps(rollout_dir, steps)
    if not all(path.is_file() for path in obs_paths):
        return None
    format_settings = _format_settings(settings)
    messages = build_messages_from_images_and_responses(
        instruction,
        obs_paths,
        [step.response for step in steps[:-1]],
        format_settings,
        target_response=steps[-1].response,
        stage_relative_paths=True,
        rollout_dir=rollout_dir,
    )
    return {
        "messages": messages,
        "task_id": infer_domain_and_task_id(rollout_dir)[1],
        "rollout_dir": str(rollout_dir),
        "step_num": steps[-1].step_num,
        "sample_mode": "full_trajectory",
        "loss_on_last_assistant_only": False,
    }


def build_holo_full_trajectory_row(
    rollout_dir: Path,
    instruction: str,
    settings: TraceScanSettings,
    rows: list[TrajRow],
) -> dict[str, Any] | None:
    agent_indices = holo_agent_row_indices(rows)
    if not agent_indices:
        return None
    converted = holo_converted_responses(rows, agent_indices)
    image_paths = holo_observation_paths(rollout_dir, rows, agent_indices, len(agent_indices) - 1)
    if not all(path.is_file() for path in image_paths):
        return None
    format_settings = _format_settings(settings)
    messages = build_messages_from_images_and_responses(
        instruction,
        image_paths,
        converted[:-1],
        format_settings,
        target_response=converted[-1],
        stage_relative_paths=True,
        rollout_dir=rollout_dir,
    )
    return {
        "messages": messages,
        "task_id": infer_domain_and_task_id(rollout_dir)[1],
        "rollout_dir": str(rollout_dir),
        "step_num": rows[agent_indices[-1]].step_num,
        "sample_mode": "full_trajectory",
        "loss_on_last_assistant_only": False,
    }


def build_rows_for_rollout(rollout_dir: Path, settings: TraceScanSettings) -> list[dict[str, Any]]:
    if settings.min_result is not None:
        score = read_result_score(rollout_dir)
        if score is None or score < settings.min_result:
            return []
    domain, task_id = infer_domain_and_task_id(rollout_dir)
    instruction = load_task_instruction(repo_path(settings.task_examples_dir), domain, task_id)
    traj_rows = load_traj_rows(rollout_dir)
    source = detect_rollout_trace_source(traj_rows, settings.trace_source)
    if settings.sample_mode == "full_trajectory":
        if source == "holo":
            row = build_holo_full_trajectory_row(rollout_dir, instruction, settings, traj_rows)
        else:
            row = build_uitars_full_trajectory_row(rollout_dir, instruction, settings)
        return [row] if row is not None else []
    if settings.sample_mode == "per_step":
        if source == "holo":
            return build_holo_per_step_rows(rollout_dir, instruction, settings, traj_rows)
        return build_uitars_per_step_rows(rollout_dir, instruction, settings)
    raise ValueError(f"Unsupported sample_mode: {settings.sample_mode}")


def build_trace_dataset_rows(settings: TraceScanSettings) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rollout_dir in iter_rollout_dirs(settings.trace_roots):
        try:
            rows.extend(build_rows_for_rollout(rollout_dir, settings))
        except (FileNotFoundError, ValueError):
            continue
    return rows
