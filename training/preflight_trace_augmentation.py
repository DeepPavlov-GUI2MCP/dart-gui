from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Literal
from urllib import error, request

import requests
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
GUI_DOCKER_ENV = REPO_ROOT / "GUI-Docker-Env"
COST_AUGMENTATION_LOG = "cost_augmentation.jsonl"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
FALLBACK_MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"prompt": 0.00000015, "completion": 0.0000006},
    "gpt-4o-mini-2024-07-18": {"prompt": 0.00000015, "completion": 0.0000006},
    "gpt-4o": {"prompt": 0.0000025, "completion": 0.00001},
    "gpt-4o-2024-08-06": {"prompt": 0.0000025, "completion": 0.00001},
}
_pricing_cache: dict[str, dict[str, float]] | None = None
if str(GUI_DOCKER_ENV) not in sys.path:
    sys.path.insert(0, str(GUI_DOCKER_ENV))

from mm_agents.env_loader import load_mm_agents_env  # noqa: E402
from mm_agents.holo.messages import observation_message, prepare_screenshot, system_message  # noqa: E402
from mm_agents.holo.schema import Step, step_json_schema  # noqa: E402

from holo_to_uitars import parse_holo_response  # noqa: E402
from uitars_trace_dataset import (  # noqa: E402
    TrajRow,
    infer_domain_and_task_id,
    is_preflight_row,
    iter_rollout_dirs,
    load_task_instruction,
    load_traj_rows,
    repo_path,
)

load_mm_agents_env()

AugmentMode = Literal["openai-api", "holo-native"]


@dataclass(frozen=True)
class TraceManifestEntry:
    domain: str
    task_id: str
    rollout_dir: str
    instruction: str


@dataclass(frozen=True)
class PreflightObservedRow:
    step_num: int
    preflight_step_index: int
    op: str
    screenshot_file: str
    action: dict[str, Any]


@dataclass(frozen=True)
class TaskMetadata:
    domain: str
    task_id: str
    instruction: str
    macro_state_id: str
    preflight_steps: list[dict[str, Any]]
    cache_key: str
    preflight_steps_hash: str


@dataclass
class CachedPreflightStep:
    preflight_step_index: int
    response: str


@dataclass
class PreflightCache:
    macro_state_id: str
    preflight_steps_hash: str
    mode: AugmentMode
    source_task_id: str
    steps: list[CachedPreflightStep] = field(default_factory=list)


@dataclass
class AugmentSettings:
    trace_root: Path
    task_examples_dir: Path
    mode: AugmentMode
    n_tasks: int = 1
    skip_first_n: int = 0
    task_ids: tuple[str, ...] = ()
    output_dir: Path | None = None
    overwrite: bool = False
    dry_run: bool = False
    write_traces_manifest: bool = False
    all_tasks: bool = False
    only_missing: bool = True


@dataclass
class TaskAugmentResult:
    task_id: str
    status: str
    source: str | None = None
    llm_calls: int = 0
    cache_hit: bool = False
    cost_usd: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class UsageCost:
    num_tokens: int
    cost_usd: float
    cost_source: str
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class CostLogContext:
    trace_root: Path
    task_id: str
    macro_state_id: str
    cache_key: str
    mode: AugmentMode


def resolve_trace_root(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def resolve_task_examples_dir(value: str | Path) -> Path:
    return repo_path(str(value))


def extract_preflight_config_steps(task_payload: dict[str, Any]) -> list[dict[str, Any]]:
    for cfg in task_payload.get("config") or []:
        if cfg.get("type") == "a11y_preflight":
            steps = (cfg.get("parameters") or {}).get("steps")
            if isinstance(steps, list):
                return steps
    return []


def load_task_metadata(task_examples_dir: Path, domain: str, task_id: str) -> TaskMetadata:
    task_path = task_examples_dir / domain / f"{task_id}.json"
    if not task_path.is_file():
        raise FileNotFoundError(f"Task example not found: {task_path}")
    with open(task_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    instruction = load_task_instruction(task_examples_dir, domain, task_id)
    synthetic = payload.get("synthetic") or {}
    macro_state_id = synthetic.get("macro_state_id")
    if not isinstance(macro_state_id, str) or not macro_state_id.strip():
        raise ValueError(f"Missing synthetic.macro_state_id in {task_path}")
    preflight_steps = extract_preflight_config_steps(payload)
    if not preflight_steps:
        raise ValueError(f"Missing a11y_preflight steps in {task_path}")
    steps_hash = hash_preflight_steps(preflight_steps)
    cache_key = f"{macro_state_id.strip()}:{steps_hash}"
    return TaskMetadata(
        domain=domain,
        task_id=task_id,
        instruction=instruction,
        macro_state_id=macro_state_id.strip(),
        preflight_steps=preflight_steps,
        cache_key=cache_key,
        preflight_steps_hash=steps_hash,
    )


def hash_preflight_steps(steps: list[dict[str, Any]]) -> str:
    normalized = json.dumps(steps, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def observed_preflight_rows(rows: list[TrajRow]) -> list[PreflightObservedRow]:
    observed: list[PreflightObservedRow] = []
    for row in rows:
        if not is_preflight_row(row):
            continue
        action = row.action
        if not isinstance(action, dict):
            continue
        index = action.get("preflight_step_index")
        op = action.get("op")
        if not isinstance(index, int) or not isinstance(op, str):
            continue
        observed.append(
            PreflightObservedRow(
                step_num=row.step_num,
                preflight_step_index=index,
                op=op,
                screenshot_file=row.screenshot_file,
                action=action,
            )
        )
    observed.sort(key=lambda item: item.preflight_step_index)
    return observed


def summarize_preflight_step(index: int, step: dict[str, Any]) -> str:
    op = step.get("op", "?")
    selector = step.get("selector") or {}
    parts = [f"step {index}: op={op}"]
    role = selector.get("role")
    name = selector.get("name")
    name_contains = selector.get("name_contains")
    if role:
        parts.append(f"role={role}")
    if name:
        parts.append(f"name={name}")
    if name_contains:
        parts.append(f"name_contains={name_contains}")
    meta = step.get("meta") or {}
    bbox = meta.get("recorded_bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        parts.append(f"recorded_bbox={bbox}")
    seconds = step.get("seconds")
    if seconds is not None:
        parts.append(f"seconds={seconds}")
    return ", ".join(parts)


def image_bytes_to_data_url(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def screenshot_dimensions(screenshot_path: Path) -> tuple[int, int]:
    with Image.open(screenshot_path) as image:
        return image.size


def bbox_center_to_holo_coords(bbox: list[Any], width: int, height: int) -> tuple[int, int]:
    x, y, w, h = [float(value) for value in bbox[:4]]
    center_x = x + w / 2.0
    center_y = y + h / 2.0
    holo_x = int(round(center_x / max(width, 1) * 1000))
    holo_y = int(round(center_y / max(height, 1) * 1000))
    return max(0, min(1000, holo_x)), max(0, min(1000, holo_y))


def element_description(step: dict[str, Any]) -> str:
    selector = step.get("selector") or {}
    meta = step.get("meta") or {}
    role = selector.get("role") or meta.get("role") or "element"
    name = selector.get("name") or meta.get("name") or selector.get("name_contains") or "target"
    return f"{role} {name}".strip()


def fallback_tool_call(step: dict[str, Any], screenshot_path: Path) -> dict[str, Any]:
    op = str(step.get("op") or "").lower()
    if op in {"click", "move"}:
        meta = step.get("meta") or {}
        bbox = meta.get("recorded_bbox")
        width, height = screenshot_dimensions(screenshot_path)
        if isinstance(bbox, list) and len(bbox) == 4:
            x, y = bbox_center_to_holo_coords(bbox, width, height)
        else:
            x, y = 500, 500
        return {
            "tool_name": "click",
            "element": element_description(step),
            "x": x,
            "y": y,
            "button": "left",
        }
    return {"tool_name": "wait"}


def expected_tool_name_for_preflight_step(step: dict[str, Any]) -> str:
    op = str(step.get("op") or "").lower()
    if op in {"click", "move"}:
        return "click"
    return "wait"


def deterministic_preflight_thought(index: int, step: dict[str, Any]) -> str:
    return f"Execute setup plan step {index + 1}: {preflight_plan_line(index, step)}"


def enforce_preflight_tool_call(
    response: str,
    *,
    step_index: int,
    config_step: dict[str, Any],
    screenshot_path: Path,
) -> str:
    payload = json.loads(response)
    tool_call = payload.get("tool_call") if isinstance(payload, dict) else None
    expected_tool = expected_tool_name_for_preflight_step(config_step)
    if not isinstance(tool_call, dict) or tool_call.get("tool_name") != expected_tool:
        payload = {
            "thought": deterministic_preflight_thought(step_index, config_step),
            "tool_call": fallback_tool_call(config_step, screenshot_path),
        }
        return Step.model_validate(payload).model_dump_json(exclude_none=True)
    return response


def normalize_step_response(
    raw_step: dict[str, Any],
    *,
    config_step: dict[str, Any],
    screenshot_path: Path,
    step_index: int = 0,
) -> str:
    thought = raw_step.get("thought")
    if not isinstance(thought, str) or not thought.strip():
        thought = deterministic_preflight_thought(step_index, config_step)
    tool_call = raw_step.get("tool_call")
    if not isinstance(tool_call, dict):
        tool_call = fallback_tool_call(config_step, screenshot_path)
    else:
        try:
            Step.model_validate({"thought": thought, "tool_call": tool_call})
        except Exception:
            tool_call = fallback_tool_call(config_step, screenshot_path)
    step = Step(thought=thought.strip(), tool_call=tool_call)
    return step.model_dump_json(exclude_none=True)


def normalize_oracle_step_response(
    raw_step: dict[str, Any],
    *,
    config_step: dict[str, Any],
    screenshot_path: Path,
    step_index: int = 0,
) -> str:
    thought = raw_step.get("thought")
    if not isinstance(thought, str) or not thought.strip():
        thought = deterministic_preflight_thought(step_index, config_step)
    step = Step(
        thought=thought.strip(),
        tool_call=fallback_tool_call(config_step, screenshot_path),
    )
    return step.model_dump_json(exclude_none=True)


def validate_response_string(response: str) -> str:
    parse_holo_response(response)
    return response


def discover_manifest_entries(trace_root: Path, task_examples_dir: Path) -> list[TraceManifestEntry]:
    entries: list[TraceManifestEntry] = []
    for rollout_dir in iter_rollout_dirs([str(trace_root)]):
        if not rollout_dir.is_relative_to(trace_root):
            continue
        domain, task_id = infer_domain_and_task_id(rollout_dir)
        try:
            instruction = load_task_instruction(task_examples_dir, domain, task_id)
        except (FileNotFoundError, ValueError):
            continue
        rel_rollout = rollout_dir.relative_to(trace_root).as_posix()
        entries.append(
            TraceManifestEntry(
                domain=domain,
                task_id=task_id,
                rollout_dir=rel_rollout,
                instruction=instruction,
            )
        )
    entries.sort(key=lambda item: (item.domain, item.task_id))
    return entries


def load_manifest(trace_root: Path, task_examples_dir: Path) -> list[TraceManifestEntry]:
    manifest_path = trace_root / "traces.jsonl"
    if manifest_path.is_file():
        entries: list[TraceManifestEntry] = []
        with open(manifest_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                entries.append(
                    TraceManifestEntry(
                        domain=str(payload["domain"]),
                        task_id=str(payload["task_id"]),
                        rollout_dir=str(payload["rollout_dir"]),
                        instruction=str(payload.get("instruction") or ""),
                    )
                )
        return entries
    discovered = discover_manifest_entries(trace_root, task_examples_dir)
    for entry in discovered:
        if not entry.instruction:
            entry = TraceManifestEntry(
                domain=entry.domain,
                task_id=entry.task_id,
                rollout_dir=entry.rollout_dir,
                instruction=load_task_instruction(task_examples_dir, entry.domain, entry.task_id),
            )
    return discovered


def write_traces_manifest(trace_root: Path, entries: list[TraceManifestEntry]) -> None:
    manifest_path = trace_root / "traces.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(
                json.dumps(
                    {
                        "domain": entry.domain,
                        "task_id": entry.task_id,
                        "rollout_dir": entry.rollout_dir,
                        "instruction": entry.instruction,
                    },
                    ensure_ascii=False,
                )
            )
            handle.write("\n")


def cache_dir_for_mode(trace_root: Path, mode: AugmentMode) -> Path:
    if mode == "holo-native":
        return trace_root / "preflight_reasoning_cache" / "holo_native"
    model = load_oracle_config()["model"]
    return trace_root / "preflight_reasoning_cache" / "openai_api" / model_path_segment(model)


def cache_path(trace_root: Path, mode: AugmentMode, cache_key: str) -> Path:
    return cache_dir_for_mode(trace_root, mode) / f"{cache_key}.json"


def load_cache(settings: AugmentSettings, cache_key: str) -> PreflightCache | None:
    path = cache_path_for_settings(settings, cache_key)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    steps = [
        CachedPreflightStep(
            preflight_step_index=int(item["preflight_step_index"]),
            response=str(item["response"]),
        )
        for item in payload.get("steps") or []
    ]
    return PreflightCache(
        macro_state_id=str(payload["macro_state_id"]),
        preflight_steps_hash=str(payload["preflight_steps_hash"]),
        mode=settings.mode,
        source_task_id=str(payload.get("source_task_id") or ""),
        steps=sorted(steps, key=lambda item: item.preflight_step_index),
    )


def save_cache(settings: AugmentSettings, cache: PreflightCache) -> None:
    path = cache_path_for_settings(settings, f"{cache.macro_state_id}:{cache.preflight_steps_hash}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "macro_state_id": cache.macro_state_id,
        "preflight_steps_hash": cache.preflight_steps_hash,
        "mode": cache.mode,
        "source_task_id": cache.source_task_id,
        "steps": [
            {
                "preflight_step_index": step.preflight_step_index,
                "response": step.response,
            }
            for step in sorted(cache.steps, key=lambda item: item.preflight_step_index)
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def cache_response_for_index(cache: PreflightCache | None, index: int) -> str | None:
    if cache is None:
        return None
    for step in cache.steps:
        if step.preflight_step_index == index:
            return step.response
    return None


def missing_cache_indices(cache: PreflightCache | None, indices: list[int]) -> list[int]:
    if cache is None:
        return indices
    present = {step.preflight_step_index for step in cache.steps}
    return [index for index in indices if index not in present]


def append_manifest_line(trace_root: Path, payload: dict[str, Any]) -> None:
    path = trace_root / "preflight_augmentation_manifest.jsonl"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False))
        handle.write("\n")


def model_path_segment(model: str) -> str:
    return model.replace("/", "__").strip()


def backend_model_for_settings(settings: AugmentSettings) -> str:
    if settings.mode == "openai-api":
        return load_oracle_config()["model"]
    return load_holo_config()["model"]


def augment_storage_dir(settings: AugmentSettings) -> Path:
    if settings.mode == "holo-native":
        return settings.trace_root / "preflight_augmented" / "holo_native"
    model = backend_model_for_settings(settings)
    return settings.trace_root / "preflight_augmented" / "openai_api" / model_path_segment(model)


def cache_storage_dir(settings: AugmentSettings) -> Path:
    if settings.mode == "holo-native":
        return settings.trace_root / "preflight_reasoning_cache" / "holo_native"
    model = backend_model_for_settings(settings)
    return settings.trace_root / "preflight_reasoning_cache" / "openai_api" / model_path_segment(model)


def cache_path_for_settings(settings: AugmentSettings, cache_key: str) -> Path:
    return cache_storage_dir(settings) / f"{cache_key}.json"


def augmented_output_dir(settings: AugmentSettings) -> Path:
    if settings.output_dir is not None:
        return settings.output_dir
    return augment_storage_dir(settings)


def augmented_task_path(settings: AugmentSettings, task_id: str) -> Path:
    return augmented_output_dir(settings) / f"{task_id}.json"


def select_manifest_entries(
    entries: list[TraceManifestEntry],
    settings: AugmentSettings,
) -> list[TraceManifestEntry]:
    if settings.task_ids:
        by_id = {entry.task_id: entry for entry in entries}
        selected: list[TraceManifestEntry] = []
        for task_id in settings.task_ids:
            entry = by_id.get(task_id)
            if entry is not None:
                selected.append(entry)
        return selected
    start = max(settings.skip_first_n, 0)
    if settings.all_tasks:
        return entries[start:]
    end = start + max(settings.n_tasks, 0)
    return entries[start:end]


def filter_entries_needing_augmentation(
    settings: AugmentSettings,
    entries: list[TraceManifestEntry],
) -> list[TraceManifestEntry]:
    if settings.overwrite or not settings.only_missing:
        return entries
    return [entry for entry in entries if not augmented_task_path(settings, entry.task_id).is_file()]


def write_augmented_preflight(
    settings: AugmentSettings,
    entry: TraceManifestEntry,
    *,
    metadata: TaskMetadata,
    observed: list[PreflightObservedRow],
    responses_by_index: dict[int, str],
    source: str,
    source_task_id: str,
) -> Path:
    output_dir = augmented_output_dir(settings)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = augmented_task_path(settings, entry.task_id)
    steps: list[dict[str, Any]] = []
    for row in observed:
        response = responses_by_index.get(row.preflight_step_index)
        if response is None:
            continue
        steps.append(
            {
                "step_num": row.step_num,
                "preflight_step_index": row.preflight_step_index,
                "response": response,
            }
        )
    payload = {
        "task_id": entry.task_id,
        "domain": entry.domain,
        "rollout_dir": entry.rollout_dir,
        "instruction": entry.instruction,
        "macro_state_id": metadata.macro_state_id,
        "cache_key": metadata.cache_key,
        "mode": settings.mode,
        "backend_model": backend_model_for_settings(settings),
        "source": source,
        "source_task_id": source_task_id,
        "steps": steps,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def load_oracle_config() -> dict[str, Any]:
    api_key = (os.getenv("PREFLIGHT_ORACLE_OPENAI_API_KEY") or "").strip()
    model = (os.getenv("PREFLIGHT_ORACLE_OPENAI_MODEL") or "").strip()
    base_url = (os.getenv("PREFLIGHT_ORACLE_OPENAI_BASE_URL") or "").strip()
    timeout = float((os.getenv("PREFLIGHT_ORACLE_TIMEOUT") or "120").strip())
    max_retries = int((os.getenv("PREFLIGHT_ORACLE_MAX_RETRIES") or "3").strip())
    missing = [name for name, value in (("PREFLIGHT_ORACLE_OPENAI_API_KEY", api_key), ("PREFLIGHT_ORACLE_OPENAI_MODEL", model)) if not value]
    if missing:
        raise ValueError(f"Missing required env vars: {', '.join(missing)}")
    return {
        "api_key": api_key,
        "model": model,
        "base_url": (base_url or "https://api.openai.com/v1").rstrip("/"),
        "timeout": timeout,
        "max_retries": max(1, max_retries),
    }


def load_holo_config() -> dict[str, Any]:
    def _ev(*names: str) -> str:
        for name in names:
            value = (os.environ.get(name) or "").strip()
            if value:
                return value
        return ""

    base_url = _ev("HOLO_OPENAI_BASE_URL", "OPENAI_BASE_URL") or "http://127.0.0.1:8010"
    api_key = _ev("HOLO_OPENAI_API_KEY", "OPENAI_API_KEY") or "empty"
    model = _ev("HOLO_OPENAI_MODEL", "OPENAI_MODEL")
    if not model:
        raise ValueError("Missing HOLO_OPENAI_MODEL or OPENAI_MODEL")
    if not base_url.rstrip("/").endswith("/v1"):
        base_url = f"{base_url.rstrip('/')}/v1"
    timeout = float((os.getenv("PREFLIGHT_HOLO_TIMEOUT") or "120").strip())
    max_retries = int((os.getenv("PREFLIGHT_HOLO_MAX_RETRIES") or "3").strip())
    return {
        "api_key": api_key,
        "model": model,
        "base_url": base_url,
        "timeout": timeout,
        "max_retries": max(1, max_retries),
        "max_pixels": float(os.getenv("PREFLIGHT_HOLO_MAX_PIXELS") or "2116800"),
        "min_pixels": float(os.getenv("PREFLIGHT_HOLO_MIN_PIXELS") or "3136"),
    }


def _normalize_model_id(model: str) -> str:
    return model.strip().lower()


def _model_lookup_keys(model: str) -> list[str]:
    normalized = _normalize_model_id(model)
    keys = [normalized]
    if "/" in normalized:
        keys.append(normalized.split("/", 1)[1])
    return keys


def fetch_openrouter_pricing(*, timeout: float = 15.0) -> dict[str, dict[str, float]]:
    global _pricing_cache
    if _pricing_cache is not None:
        return _pricing_cache

    pricing: dict[str, dict[str, float]] = dict(FALLBACK_MODEL_PRICING)
    try:
        req = request.Request(OPENROUTER_MODELS_URL, method="GET")
        with request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (error.URLError, error.HTTPError, json.JSONDecodeError, TimeoutError):
        _pricing_cache = pricing
        return pricing

    for entry in body.get("data", []):
        model_id = entry.get("id")
        raw_pricing = entry.get("pricing") or {}
        prompt = raw_pricing.get("prompt")
        completion = raw_pricing.get("completion")
        if not model_id or prompt is None or completion is None:
            continue
        rates = {"prompt": float(prompt), "completion": float(completion)}
        pricing[_normalize_model_id(model_id)] = rates
        if "/" in model_id:
            pricing[_normalize_model_id(model_id.split("/", 1)[1])] = rates

    _pricing_cache = pricing
    return pricing


def lookup_model_rates(model: str) -> dict[str, float]:
    pricing = fetch_openrouter_pricing()
    for key in _model_lookup_keys(model):
        rates = pricing.get(key)
        if rates is not None:
            return rates
    for key in _model_lookup_keys(model):
        for known, rates in pricing.items():
            if key in known or known in key:
                return rates
    raise KeyError(f"no pricing found for model: {model}")


def is_local_inference_url(base_url: str) -> bool:
    lowered = base_url.lower()
    return "127.0.0.1" in lowered or "localhost" in lowered


def extract_api_cost_usd(response_body: dict[str, Any]) -> float | None:
    usage = response_body.get("usage") or {}
    for container in (usage, response_body):
        cost = container.get("cost")
        if cost is not None:
            return float(cost)
        total_cost = container.get("total_cost")
        if total_cost is not None:
            return float(total_cost)
    return None


def compute_usage_cost(
    *,
    response_body: dict[str, Any],
    model: str,
    base_url: str,
) -> UsageCost:
    usage = response_body.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    num_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))

    if is_local_inference_url(base_url):
        return UsageCost(
            num_tokens=num_tokens,
            cost_usd=0.0,
            cost_source="local_zero",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    api_cost = extract_api_cost_usd(response_body)
    if api_cost is not None:
        return UsageCost(
            num_tokens=num_tokens,
            cost_usd=api_cost,
            cost_source="api",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    try:
        rates = lookup_model_rates(model)
    except KeyError:
        return UsageCost(
            num_tokens=num_tokens,
            cost_usd=0.0,
            cost_source="unknown_zero",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    cost_usd = (prompt_tokens * rates["prompt"]) + (completion_tokens * rates["completion"])
    return UsageCost(
        num_tokens=num_tokens,
        cost_usd=cost_usd,
        cost_source="openrouter_pricing",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def append_cost_augmentation_log(trace_root: Path, record: dict[str, Any]) -> Path:
    path = trace_root / COST_AUGMENTATION_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def log_augmentation_cost(
    cost_context: CostLogContext,
    *,
    response_body: dict[str, Any],
    model: str,
    base_url: str,
    preflight_step_indices: list[int],
) -> UsageCost:
    usage_cost = compute_usage_cost(response_body=response_body, model=model, base_url=base_url)
    append_cost_augmentation_log(
        cost_context.trace_root,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": cost_context.task_id,
            "macro_state_id": cost_context.macro_state_id,
            "cache_key": cost_context.cache_key,
            "mode": cost_context.mode,
            "preflight_step_indices": preflight_step_indices,
            "model": model,
            "base_url": base_url,
            "num_tokens": usage_cost.num_tokens,
            "prompt_tokens": usage_cost.prompt_tokens,
            "completion_tokens": usage_cost.completion_tokens,
            "cost": usage_cost.cost_usd,
            "cost_source": usage_cost.cost_source,
        },
    )
    return usage_cost


def extract_completion_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"Unexpected completion payload: {payload!r}")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(part.strip() for part in parts if part.strip()).strip()
    return str(content).strip()


def post_chat_completion(config: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }
    endpoint = config["base_url"].rstrip("/") + "/chat/completions"
    last_error: Exception | None = None
    for _ in range(config["max_retries"]):
        try:
            response = requests.post(endpoint, headers=headers, json=body, timeout=config["timeout"])
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Chat completion request failed: {last_error}") from last_error


def parse_json_response_text(response_text: str) -> Any:
    cleaned = response_text.strip()
    candidates = [cleaned]
    if "```" in cleaned:
        for chunk in cleaned.split("```"):
            chunk = chunk.strip()
            if not chunk:
                continue
            if chunk.startswith("json"):
                chunk = chunk[4:].strip()
            candidates.append(chunk)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not parse JSON response: {response_text!r}")


def build_oracle_messages(
    metadata: TaskMetadata,
    observed: list[PreflightObservedRow],
    rollout_dir: Path,
    indices: list[int],
) -> list[dict[str, Any]]:
    summaries = [
        summarize_preflight_step(index, metadata.preflight_steps[index])
        for index in indices
        if index < len(metadata.preflight_steps)
    ]
    system_text = (
        "You are a meta-observer annotating a GUI agent trace. "
        "Given screenshots and a fixed preflight setup script, write plausible Holotron-style "
        "reasoning for each preflight step. Preflight setup happens before the main task. "
        "Actions are scripted separately; provide thought only. "
        "Return JSON only: {\"steps\":[{\"preflight_step_index\":0,\"thought\":\"...\"}]}"
    )
    user_parts: list[Any] = [
        {
            "type": "text",
            "text": (
                f"Main task instruction (context only): {metadata.instruction}\n\n"
                f"Preflight setup script ({len(indices)} steps to annotate):\n"
                + "\n".join(f"- {summary}" for summary in summaries)
            ),
        }
    ]
    initial_rows = [row for row in load_traj_rows(rollout_dir) if row.step_num == 1]
    if initial_rows:
        initial_path = rollout_dir / initial_rows[0].screenshot_file
        if initial_path.is_file():
            user_parts.append({"type": "text", "text": "Initial desktop screenshot:"})
            user_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_bytes_to_data_url(initial_path.read_bytes())},
                }
            )
    for index in indices:
        row = next(item for item in observed if item.preflight_step_index == index)
        screenshot_path = rollout_dir / row.screenshot_file
        if not screenshot_path.is_file():
            raise FileNotFoundError(f"Missing screenshot: {screenshot_path}")
        user_parts.append(
            {
                "type": "text",
                "text": (
                    f"Preflight step {index} ({row.op}) pre-action screenshot: "
                    f"{summarize_preflight_step(index, metadata.preflight_steps[index])}"
                ),
            }
        )
        user_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": image_bytes_to_data_url(screenshot_path.read_bytes())},
            }
        )
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_parts},
    ]


def generate_oracle_responses(
    metadata: TaskMetadata,
    observed: list[PreflightObservedRow],
    rollout_dir: Path,
    indices: list[int],
    config: dict[str, Any],
    *,
    cost_context: CostLogContext | None = None,
) -> tuple[dict[int, str], float]:
    messages = build_oracle_messages(metadata, observed, rollout_dir, indices)
    body = {
        "model": config["model"],
        "response_format": {"type": "json_object"},
        "messages": messages,
    }
    payload = post_chat_completion(config, body)
    cost_usd = 0.0
    if cost_context is not None:
        usage_cost = log_augmentation_cost(
            cost_context,
            response_body=payload,
            model=config["model"],
            base_url=config["base_url"],
            preflight_step_indices=indices,
        )
        cost_usd = usage_cost.cost_usd
    parsed = parse_json_response_text(extract_completion_text(payload))
    steps = parsed.get("steps") if isinstance(parsed, dict) else parsed
    if not isinstance(steps, list):
        raise ValueError(f"Oracle response missing steps list: {parsed!r}")
    responses: dict[int, str] = {}
    for raw_step in steps:
        if not isinstance(raw_step, dict):
            continue
        index = raw_step.get("preflight_step_index")
        if not isinstance(index, int):
            continue
        row = next((item for item in observed if item.preflight_step_index == index), None)
        if row is None or index >= len(metadata.preflight_steps):
            continue
        screenshot_path = rollout_dir / row.screenshot_file
        responses[index] = validate_response_string(
            normalize_oracle_step_response(
                raw_step,
                config_step=metadata.preflight_steps[index],
                screenshot_path=screenshot_path,
                step_index=index,
            )
        )
    for index in indices:
        if index in responses:
            continue
        row = next(item for item in observed if item.preflight_step_index == index)
        screenshot_path = rollout_dir / row.screenshot_file
        config_step = metadata.preflight_steps[index]
        fallback = Step(
            thought=deterministic_preflight_thought(index, config_step),
            tool_call=fallback_tool_call(config_step, screenshot_path),
        )
        responses[index] = fallback.model_dump_json(exclude_none=True)
    return responses, cost_usd


def build_preflight_system_prompt(metadata: TaskMetadata) -> str:
    schema_block = json.dumps(step_json_schema(), indent=2)
    return (
        "You are a GUI agent controlling an Ubuntu desktop. "
        "Your task is to execute the setup plan exactly, one step at a time. "
        "The user will provide the current screenshot, the full setup plan, and the current plan step. "
        "Emit the single next Holo JSON step for the current plan step only. "
        "Coordinates are integers in [0, 1000].\n\n"
        f"<output_format>\n```json\n{schema_block}\n```\n</output_format>"
    )


def preflight_plan_line(index: int, step: dict[str, Any]) -> str:
    op = str(step.get("op") or "")
    selector = step.get("selector") or {}
    meta = step.get("meta") or {}
    name = selector.get("name") or meta.get("name")
    role = selector.get("role") or meta.get("role")
    name_contains = selector.get("name_contains")
    if op == "wait_for":
        target = name or name_contains or role or "the requested UI"
        return f"{index + 1}. Wait until {target} is visible."
    if op == "assert":
        target = name or name_contains or role or "the requested UI"
        return f"{index + 1}. Confirm {target} is visible and wait."
    if op == "sleep":
        return f"{index + 1}. Wait for the interface to settle."
    if op in {"click", "move"}:
        target = " ".join(part for part in (role, name or name_contains) if part) or "the target UI element"
        return f"{index + 1}. Click {target}."
    return f"{index + 1}. Execute setup operation: {summarize_preflight_step(index, step)}."


def build_preflight_setup_plan(metadata: TaskMetadata) -> str:
    return "\n".join(
        preflight_plan_line(index, step)
        for index, step in enumerate(metadata.preflight_steps)
    )


def build_preflight_user_text(
    metadata: TaskMetadata,
    *,
    step_index: int,
    completed_indices: list[int],
) -> str:
    plan = build_preflight_setup_plan(metadata)
    current = preflight_plan_line(step_index, metadata.preflight_steps[step_index])
    completed_lines = [
        f"- {preflight_plan_line(index, metadata.preflight_steps[index])}"
        for index in completed_indices
    ]
    completed_block = "\n".join(completed_lines) if completed_lines else "- none"
    return (
        "Setup plan:\n"
        f"{plan}\n\n"
        f"Completed preflight steps:\n{completed_block}"
        f"\n\nCurrent plan step:\n{current}"
    )


def generate_holo_native_response(
    metadata: TaskMetadata,
    rollout_dir: Path,
    row: PreflightObservedRow,
    *,
    completed_indices: list[int],
    config: dict[str, Any],
    cost_context: CostLogContext | None = None,
) -> tuple[str, float]:
    screenshot_path = rollout_dir / row.screenshot_file
    if not screenshot_path.is_file():
        raise FileNotFoundError(f"Missing screenshot: {screenshot_path}")
    image, _, _, _, _ = prepare_screenshot(
        screenshot_path.read_bytes(),
        max_pixels=config["max_pixels"],
        min_pixels=config["min_pixels"],
    )
    user_text = build_preflight_user_text(
        metadata,
        step_index=row.preflight_step_index,
        completed_indices=completed_indices,
    )
    messages = [
        system_message(build_preflight_system_prompt(metadata)),
        observation_message(image),
        {"role": "user", "content": user_text},
    ]
    body = {
        "model": config["model"],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "holo_step",
                "schema": step_json_schema(),
                "strict": True,
            },
        },
        "messages": messages,
    }
    payload = post_chat_completion(config, body)
    cost_usd = 0.0
    if cost_context is not None:
        usage_cost = log_augmentation_cost(
            cost_context,
            response_body=payload,
            model=config["model"],
            base_url=config["base_url"],
            preflight_step_indices=[row.preflight_step_index],
        )
        cost_usd = usage_cost.cost_usd
    raw = parse_json_response_text(extract_completion_text(payload))
    if not isinstance(raw, dict):
        raise ValueError(f"Holo-native response must be an object: {raw!r}")
    normalized = validate_response_string(
        normalize_step_response(
            raw,
            config_step=metadata.preflight_steps[row.preflight_step_index],
            screenshot_path=screenshot_path,
            step_index=row.preflight_step_index,
        )
    )
    return enforce_preflight_tool_call(
        normalized,
        step_index=row.preflight_step_index,
        config_step=metadata.preflight_steps[row.preflight_step_index],
        screenshot_path=screenshot_path,
    ), cost_usd


def merge_cache_steps(cache: PreflightCache | None, new_steps: dict[int, str]) -> PreflightCache:
    merged: dict[int, str] = {}
    if cache is not None:
        for step in cache.steps:
            merged[step.preflight_step_index] = step.response
    merged.update(new_steps)
    return PreflightCache(
        macro_state_id=cache.macro_state_id if cache else "",
        preflight_steps_hash=cache.preflight_steps_hash if cache else "",
        mode=cache.mode if cache else "openai-api",
        source_task_id=cache.source_task_id if cache else "",
        steps=[
            CachedPreflightStep(preflight_step_index=index, response=response)
            for index, response in sorted(merged.items())
        ],
    )


def augment_rollout(settings: AugmentSettings, entry: TraceManifestEntry) -> TaskAugmentResult:
    rollout_dir = settings.trace_root / entry.rollout_dir
    output_path = augmented_task_path(settings, entry.task_id)
    if output_path.is_file() and not settings.overwrite:
        return TaskAugmentResult(task_id=entry.task_id, status="skipped", source="exists")

    rows = load_traj_rows(rollout_dir)
    observed = observed_preflight_rows(rows)
    if not observed:
        return TaskAugmentResult(task_id=entry.task_id, status="skipped", error="no preflight rows")

    metadata = load_task_metadata(settings.task_examples_dir, entry.domain, entry.task_id)
    indices = [row.preflight_step_index for row in observed]
    cache = load_cache(settings, metadata.cache_key)

    missing = missing_cache_indices(cache, indices)
    responses_by_index: dict[int, str] = {}
    for index in indices:
        cached = cache_response_for_index(cache, index)
        if cached is not None and index not in missing:
            responses_by_index[index] = cached

    source = "cache"
    llm_calls = 0
    cost_usd = 0.0

    if missing and not settings.dry_run:
        cost_context = CostLogContext(
            trace_root=settings.trace_root,
            task_id=entry.task_id,
            macro_state_id=metadata.macro_state_id,
            cache_key=metadata.cache_key,
            mode=settings.mode,
        )
        if settings.mode == "openai-api":
            config = load_oracle_config()
            generated, call_cost = generate_oracle_responses(
                metadata,
                observed,
                rollout_dir,
                missing,
                config,
                cost_context=cost_context,
            )
            llm_calls = 1
            cost_usd += call_cost
            source = "llm" if len(responses_by_index) == 0 else "cache+llm"
        else:
            config = load_holo_config()
            generated = {}
            completed: list[int] = []
            for index in sorted(missing):
                row = next(item for item in observed if item.preflight_step_index == index)
                response, call_cost = generate_holo_native_response(
                    metadata,
                    rollout_dir,
                    row,
                    completed_indices=completed,
                    config=config,
                    cost_context=cost_context,
                )
                generated[index] = response
                completed.append(index)
                llm_calls += 1
                cost_usd += call_cost
            source = "llm" if len(responses_by_index) == 0 else "cache+llm"

        responses_by_index.update(generated)
        base_cache = cache or PreflightCache(
            macro_state_id=metadata.macro_state_id,
            preflight_steps_hash=metadata.preflight_steps_hash,
            mode=settings.mode,
            source_task_id=entry.task_id,
            steps=[],
        )
        merged = merge_cache_steps(base_cache, generated)
        merged.source_task_id = entry.task_id
        save_cache(settings, merged)
        cache = merged
    elif missing and settings.dry_run:
        source = "dry-run"

    if not missing:
        source = "cache"

    for index in indices:
        cached = cache_response_for_index(cache, index)
        if cached is not None:
            responses_by_index[index] = cached

    if settings.dry_run:
        return TaskAugmentResult(
            task_id=entry.task_id,
            status="dry-run",
            source=source,
            llm_calls=llm_calls,
            cache_hit=not missing,
            cost_usd=cost_usd,
        )

    output_path = write_augmented_preflight(
        settings,
        entry,
        metadata=metadata,
        observed=observed,
        responses_by_index=responses_by_index,
        source=source,
        source_task_id=cache.source_task_id if cache else entry.task_id,
    )
    try:
        output_rel = str(output_path.relative_to(settings.trace_root))
    except ValueError:
        output_rel = str(output_path)
    append_manifest_line(
        settings.trace_root,
        {
            "task_id": entry.task_id,
            "domain": entry.domain,
            "macro_state_id": metadata.macro_state_id,
            "cache_key": metadata.cache_key,
            "mode": settings.mode,
            "backend_model": backend_model_for_settings(settings),
            "source": source,
            "source_task_id": cache.source_task_id if cache else entry.task_id,
            "rollout_dir": entry.rollout_dir,
            "output_path": output_rel,
        },
    )
    return TaskAugmentResult(
        task_id=entry.task_id,
        status="ok",
        source=source,
        llm_calls=llm_calls,
        cache_hit=not bool(missing),
        cost_usd=cost_usd,
    )


def run_augmentation(settings: AugmentSettings) -> dict[str, Any]:
    settings.trace_root.mkdir(parents=True, exist_ok=True)
    if settings.overwrite:
        cache_root = cache_storage_dir(settings)
        if cache_root.is_dir():
            shutil.rmtree(cache_root)
    if settings.write_traces_manifest:
        entries = discover_manifest_entries(settings.trace_root, settings.task_examples_dir)
        write_traces_manifest(settings.trace_root, entries)
    else:
        entries = load_manifest(settings.trace_root, settings.task_examples_dir)
    selected = select_manifest_entries(entries, settings)
    pending = filter_entries_needing_augmentation(settings, selected)

    results: list[TaskAugmentResult] = []
    failures: list[dict[str, str]] = []
    cache_hits = 0
    llm_calls = 0
    total_cost = 0.0
    progress = tqdm(
        pending,
        desc=f"Preflight augment [{settings.trace_root.name}]",
        unit="task",
        dynamic_ncols=True,
    )
    for entry in progress:
        try:
            result = augment_rollout(settings, entry)
            results.append(result)
            if result.cache_hit:
                cache_hits += 1
            llm_calls += result.llm_calls
            total_cost += result.cost_usd
            progress.set_postfix(
                ok=sum(1 for item in results if item.status == "ok"),
                skip=sum(1 for item in results if item.status == "skipped"),
                fail=len(failures),
                refresh=False,
            )
        except Exception as exc:
            failures.append({"task_id": entry.task_id, "error": str(exc)})

    processed = sum(1 for item in results if item.status in {"ok", "dry-run"})
    skipped = sum(1 for item in results if item.status == "skipped")
    return {
        "ok": not failures,
        "trace_root": str(settings.trace_root),
        "output_dir": str(augmented_output_dir(settings)),
        "cache_dir": str(cache_storage_dir(settings)),
        "mode": settings.mode,
        "backend_model": backend_model_for_settings(settings),
        "selected_tasks": len(selected),
        "pending_tasks": len(pending),
        "processed": processed,
        "skipped": skipped,
        "cache_hits": cache_hits,
        "llm_calls": llm_calls,
        "total_cost": total_cost,
        "cost_log": str(settings.trace_root / COST_AUGMENTATION_LOG),
        "results": [
            {
                "task_id": item.task_id,
                "status": item.status,
                "source": item.source,
                "llm_calls": item.llm_calls,
                "cache_hit": item.cache_hit,
                "cost_usd": item.cost_usd,
                "error": item.error,
            }
            for item in results
        ],
        "failures": failures,
    }
