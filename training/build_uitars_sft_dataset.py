from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from uitars_format import UitarsFormatSettings
from uitars_trace_dataset import TraceScanSettings, build_trace_dataset_rows, repo_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and cache UI-TARS trace SFT JSONL.")
    parser.add_argument("--config", type=str, required=True, help="Path to train_sft YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Print summary without writing JSONL.")
    parser.add_argument("--refresh", action="store_true", help="Force rebuild even if cache exists.")
    return parser


def get_mapping(root: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = root.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"`{key}` must be a YAML object.")
    return dict(value)


def get_optional_str(root: Mapping[str, Any], key: str, default: str | None = None) -> str | None:
    value = root.get(key, default)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"`{key}` must be a string.")
    stripped = value.strip()
    return stripped if stripped else default


def get_optional_int(root: Mapping[str, Any], key: str, default: int) -> int:
    value = root.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"`{key}` must be an integer.")
    return value


def get_optional_float(root: Mapping[str, Any], key: str) -> float | None:
    value = root.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"`{key}` must be a number.")
    return float(value)


def get_optional_bool(root: Mapping[str, Any], key: str, default: bool) -> bool:
    value = root.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"`{key}` must be a boolean.")
    return value


def load_trace_scan_settings(config_path: str) -> tuple[TraceScanSettings, Path, bool]:
    path = Path(config_path).expanduser().resolve()
    with open(path, encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    root = payload.get("train_sft", payload)
    dataset_cfg = get_mapping(root, "dataset")
    output_cfg = get_mapping(root, "output")
    dataset_format = get_optional_str(dataset_cfg, "format", "chat") or "chat"
    if dataset_format != "uitars_trace":
        raise ValueError("`dataset.format` must be `uitars_trace` for this script.")

    trace_roots_raw = dataset_cfg.get("trace_roots")
    if trace_roots_raw is None:
        trace_roots_raw = []
    if not isinstance(trace_roots_raw, list):
        raise ValueError("`dataset.trace_roots` must be a list.")
    trace_roots = tuple(str(item).strip() for item in trace_roots_raw if str(item).strip())
    preflight_augmented_path = get_optional_str(dataset_cfg, "preflight_augmented_path")
    if not trace_roots and not preflight_augmented_path:
        raise ValueError("`dataset.trace_roots` or `dataset.preflight_augmented_path` is required.")
    task_examples_dir = get_optional_str(dataset_cfg, "task_examples_dir")
    if not task_examples_dir:
        raise ValueError("`dataset.task_examples_dir` is required for uitars_trace datasets.")

    uitars_cfg = get_mapping(dataset_cfg, "uitars")
    history_n = get_optional_int(dataset_cfg, "history_n", 5)
    uitars = UitarsFormatSettings(
        prompt_style=get_optional_str(uitars_cfg, "prompt_style", "qwen25vl_normal") or "qwen25vl_normal",
        infer_mode=get_optional_str(uitars_cfg, "infer_mode", "qwen25vl_normal") or "qwen25vl_normal",
        language=get_optional_str(uitars_cfg, "language", "English") or "English",
        max_pixels=get_optional_int(uitars_cfg, "max_pixels", 16384 * 28 * 28),
        min_pixels=get_optional_int(uitars_cfg, "min_pixels", 100 * 28 * 28),
        history_n=history_n,
    )
    trace_source = get_optional_str(dataset_cfg, "trace_source", "auto") or "auto"
    if trace_source not in {"auto", "uitars", "holo"}:
        raise ValueError("`dataset.trace_source` must be one of: auto, uitars, holo.")

    trace_root = get_optional_str(dataset_cfg, "trace_root")
    max_rollouts_raw = dataset_cfg.get("max_rollouts")
    max_rollouts: int | None = None
    if max_rollouts_raw is not None:
        if isinstance(max_rollouts_raw, bool) or not isinstance(max_rollouts_raw, int):
            raise ValueError("`dataset.max_rollouts` must be an integer.")
        max_rollouts = max_rollouts_raw

    max_preflight_steps_raw = dataset_cfg.get("max_preflight_steps")
    max_preflight_steps: int | None = None
    if max_preflight_steps_raw is not None:
        if isinstance(max_preflight_steps_raw, bool) or not isinstance(max_preflight_steps_raw, int):
            raise ValueError("`dataset.max_preflight_steps` must be an integer.")
        max_preflight_steps = max_preflight_steps_raw

    settings = TraceScanSettings(
        trace_roots=trace_roots,
        task_examples_dir=task_examples_dir,
        sample_mode=get_optional_str(dataset_cfg, "sample_mode", "per_step") or "per_step",
        history_n=history_n,
        min_result=get_optional_float(dataset_cfg, "min_result"),
        trace_source=trace_source,  # type: ignore[arg-type]
        uitars=uitars,
        preflight_augmented_path=preflight_augmented_path,
        trace_root=trace_root,
        max_rollouts=max_rollouts,
        max_preflight_steps=max_preflight_steps,
    )
    datasets_dir = repo_path(get_optional_str(output_cfg, "datasets_dir", "datasets/runtime") or "datasets/runtime")
    refresh = get_optional_bool(dataset_cfg, "refresh", False)
    return settings, datasets_dir, refresh


def cache_name(settings: TraceScanSettings) -> str:
    raw = json.dumps(
        {
            "trace_roots": settings.trace_roots,
            "task_examples_dir": settings.task_examples_dir,
            "sample_mode": settings.sample_mode,
            "history_n": settings.history_n,
            "min_result": settings.min_result,
            "trace_source": settings.trace_source,
            "preflight_augmented_path": settings.preflight_augmented_path,
            "trace_root": settings.trace_root,
            "max_rollouts": settings.max_rollouts,
            "max_preflight_steps": settings.max_preflight_steps,
            "uitars": {
                "prompt_style": settings.uitars.prompt_style,
                "infer_mode": settings.uitars.infer_mode,
                "language": settings.uitars.language,
                "max_pixels": settings.uitars.max_pixels,
                "min_pixels": settings.uitars.min_pixels,
            },
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"uitars-traces-{digest}"


def stage_trace_dataset(settings: TraceScanSettings, datasets_dir: Path, *, refresh: bool) -> Path:
    dataset_dir = datasets_dir / cache_name(settings)
    data_path = dataset_dir / "data.jsonl"
    if data_path.is_file() and not refresh:
        return data_path
    rows = build_trace_dataset_rows(settings)
    if not rows:
        raise RuntimeError("No UI-TARS trace rows were built. Check trace_roots and task_examples_dir.")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with open(data_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return data_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    settings, datasets_dir, config_refresh = load_trace_scan_settings(args.config)
    if args.dry_run:
        rows = build_trace_dataset_rows(settings)
        print(f"Would stage {len(rows)} rows under {datasets_dir / cache_name(settings) / 'data.jsonl'}")
        return 0
    data_path = stage_trace_dataset(settings, datasets_dir, refresh=config_refresh or args.refresh)
    print(f"Staged dataset: {data_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
