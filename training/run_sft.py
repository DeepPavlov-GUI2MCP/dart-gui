from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

import yaml

from build_uitars_sft_dataset import cache_name as uitars_cache_name
from build_uitars_sft_dataset import stage_trace_dataset
from hf_hub import configure_hf_hub, load_pretrained
from mlflow_utils import MlflowSettings, build_mlflow_callback, configure_mlflow
from sft_callbacks import LoraEpochCheckpointCallback
from uitars_collator import (
    PretokenizedUitarsDataset,
    UitarsTraceDataset,
    make_pretokenized_collator,
    make_uitars_data_collator,
)
from uitars_format import UitarsFormatSettings
from uitars_trace_dataset import TraceScanSettings, build_trace_dataset_rows


REPO_ROOT = Path(__file__).resolve().parents[1]
_LOG_HANDLES: list[TextIO] = []

VISION_MODEL_TYPES = frozenset(
    {
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3_vl",
        "llava",
        "llava_next",
        "llava_onevision",
        "mllama",
        "paligemma",
        "idefics2",
        "idefics3",
        "internvl",
        "smolvlm",
    }
)


@dataclass(frozen=True)
class DatasetSettings:
    format: str = "chat"
    repo_id: str | None = None
    split: str = "train"
    name: str | None = None
    revision: str | None = None
    data_files: Any = None
    trust_remote_code: bool = False
    field_messages: str = "messages"
    message_property_mappings: dict[str, str] = field(
        default_factory=lambda: {"role": "role", "content": "content"}
    )
    refresh: bool = False
    trace_roots: tuple[str, ...] = ()
    task_examples_dir: str | None = None
    sample_mode: str = "per_step"
    history_n: int = 5
    min_result: float | None = None
    trace_source: str = "auto"
    preflight_augmented_path: str | None = None
    trace_root: str | None = None
    max_rollouts: int | None = None
    max_preflight_steps: int | None = None
    goal_variants: bool = False
    max_goal_variants: int = 1
    task_generation_root: str | None = None
    pretokenized_traces_dir: str | None = None
    pretokenized_split: str = "train"
    smoke_microactions: int | None = None
    uitars: UitarsFormatSettings = field(default_factory=UitarsFormatSettings)


@dataclass(frozen=True)
class ModelSettings:
    base_model: str
    max_seq_length: int = 2048
    trust_remote_code: bool = True


@dataclass(frozen=True)
class LoraSettings:
    r: int = 16
    alpha: int = 32
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "down_proj",
        "up_proj",
    )
    dropout: float = 0.0


@dataclass(frozen=True)
class TrainingSettings:
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    warmup_steps: int = 10
    logging_steps: int = 1
    max_steps: int | None = None
    num_train_epochs: int = 1
    save_steps: int = 500
    weight_decay: float = 0.0
    packing: bool = True
    gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = True
    ddp_find_unused_parameters: bool = False
    dataloader_num_workers: int | None = None
    dataloader_prefetch_factor: int | None = None
    dataloader_persistent_workers: bool = False
    save_lora_every_n_epochs: float = 1.0
    eval_every_n_epochs: float = 0.5


@dataclass(frozen=True)
class HuggingFaceSettings:
    api_key: str | None = None


@dataclass(frozen=True)
class HubSettings:
    push_to_hub: bool = True
    model_id: str | None = None
    private: bool = True
    revision: str | None = None


@dataclass(frozen=True)
class ResolvedConfig:
    config_path: str | None
    config_mode: bool
    model: ModelSettings
    output_dir: Path
    datasets_dir: Path
    dataset: DatasetSettings
    lora: LoraSettings
    training: TrainingSettings
    hf: HuggingFaceSettings
    hub: HubSettings
    mlflow: MlflowSettings = field(default_factory=MlflowSettings)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train models with TRL SFT (LoRA).")
    parser.add_argument("--config", type=str, default=None, help="Path to a full YAML config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and stage data without loading model or training.",
    )
    parser.add_argument(
        "--skip-hf-validation",
        action="store_true",
        help="Skip Hub write validation for dry-run debugging.",
    )
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--dataset_repo", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hf_api_key", type=str, default=None)
    parser.add_argument("--no_push_to_hub", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="outputs/sft")
    parser.add_argument(
        "--from-pretokenized-traces",
        type=str,
        default=None,
        help="Train from pretokenized shard directory instead of staging/tokenizing traces.",
    )
    return parser


def reject_mixed_config_usage(parser: argparse.ArgumentParser, argv: Sequence[str], args: argparse.Namespace) -> None:
    if not args.config:
        return
    allowed = {"--config", "--dry-run", "--skip-hf-validation", "--from-pretokenized-traces"}
    option_names = [token.split("=", 1)[0] for token in argv if token.startswith("--")]
    disallowed = [name for name in option_names if name not in allowed]
    if disallowed:
        parser.error("`--config` cannot be combined with training overrides. Use either config mode or args mode.")


def load_config_file(path: str) -> ResolvedConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(config_path, encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a YAML object: {path}")
    root = payload.get("train_sft", payload)
    if not isinstance(root, dict):
        raise ValueError(f"`train_sft` config must be a YAML object: {path}")
    return resolve_config(root, config_path=str(config_path), config_mode=True)


def resolve_args(args: argparse.Namespace) -> ResolvedConfig:
    if not args.base_model or not args.dataset_repo:
        raise ValueError("args mode requires `--base_model` and `--dataset_repo`.")
    root = {
        "model": {"base_model": args.base_model},
        "dataset": {"repo_id": args.dataset_repo, "split": args.dataset_split},
        "output": {"output_dir": args.output_dir},
        "hf": {"api_key": args.hf_api_key},
        "hub": {"push_to_hub": not args.no_push_to_hub, "model_id": args.hub_model_id},
    }
    return resolve_config(root, config_path=None, config_mode=False)


def resolve_config(root: Mapping[str, Any], *, config_path: str | None, config_mode: bool) -> ResolvedConfig:
    model_cfg = get_mapping(root, "model")
    dataset_cfg = get_mapping(root, "dataset")
    output_cfg = get_mapping(root, "output")
    hf_cfg = get_mapping(root, "hf")
    hub_cfg = get_mapping(root, "hub")
    lora_cfg = get_mapping(root, "lora")
    training_cfg = get_mapping(root, "training")

    model = ModelSettings(
        base_model=get_required_str(model_cfg, "base_model", "model"),
        max_seq_length=get_optional_int(model_cfg, "max_seq_length", 2048),
        trust_remote_code=get_optional_bool(model_cfg, "trust_remote_code", True),
    )

    dataset_format = get_optional_str(dataset_cfg, "format", "chat") or "chat"
    trace_roots_raw = dataset_cfg.get("trace_roots", [])
    if trace_roots_raw is None:
        trace_roots_raw = []
    if not isinstance(trace_roots_raw, list):
        raise ValueError("`dataset.trace_roots` must be a list.")
    trace_roots = tuple(str(item).strip() for item in trace_roots_raw if str(item).strip())

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

    min_result_raw = dataset_cfg.get("min_result")
    min_result: float | None = None
    if min_result_raw is not None:
        if isinstance(min_result_raw, bool) or not isinstance(min_result_raw, (int, float)):
            raise ValueError("`dataset.min_result` must be a number.")
        min_result = float(min_result_raw)

    trace_source = get_optional_str(dataset_cfg, "trace_source", "auto") or "auto"
    if trace_source not in {"auto", "uitars", "holo"}:
        raise ValueError("`dataset.trace_source` must be one of: auto, uitars, holo.")

    preflight_augmented_path = get_optional_str(dataset_cfg, "preflight_augmented_path")
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

    smoke_microactions_raw = dataset_cfg.get("smoke_microactions")
    smoke_microactions: int | None = None
    if smoke_microactions_raw is not None:
        if isinstance(smoke_microactions_raw, bool) or not isinstance(smoke_microactions_raw, int):
            raise ValueError("`dataset.smoke_microactions` must be an integer.")
        smoke_microactions = smoke_microactions_raw

    goal_variants = get_optional_bool(dataset_cfg, "goal_variants", False)
    max_goal_variants = get_optional_int(dataset_cfg, "max_goal_variants", 1)
    task_generation_root = get_optional_str(dataset_cfg, "task_generation_root")
    pretokenized_traces_dir = get_optional_str(dataset_cfg, "pretokenized_traces_dir")
    pretokenized_split = get_optional_str(dataset_cfg, "pretokenized_split", "train") or "train"
    if pretokenized_split == "dev":
        pretokenized_split = "val"
    if pretokenized_split not in {"train", "val", "all"}:
        raise ValueError("`dataset.pretokenized_split` must be one of: train, val, all.")
    if goal_variants and not task_generation_root:
        raise ValueError("`dataset.task_generation_root` is required when `dataset.goal_variants` is true.")

    repo_id = get_optional_str(dataset_cfg, "repo_id")
    if dataset_format == "chat":
        if not repo_id:
            raise ValueError("`dataset.repo_id` is required when `dataset.format` is `chat`.")
    elif dataset_format == "uitars_trace":
        if not trace_roots and not preflight_augmented_path:
            raise ValueError(
                "`dataset.trace_roots` or `dataset.preflight_augmented_path` is required "
                "when `dataset.format` is `uitars_trace`."
            )
        if not get_optional_str(dataset_cfg, "task_examples_dir"):
            raise ValueError("`dataset.task_examples_dir` is required when `dataset.format` is `uitars_trace`.")
    else:
        raise ValueError("`dataset.format` must be `chat` or `uitars_trace`.")

    resolved_dataset = DatasetSettings(
        format=dataset_format,
        repo_id=repo_id,
        split=get_optional_str(dataset_cfg, "split", "train") or "train",
        name=get_optional_str(dataset_cfg, "name"),
        revision=get_optional_str(dataset_cfg, "revision"),
        data_files=dataset_cfg.get("data_files"),
        trust_remote_code=get_optional_bool(dataset_cfg, "trust_remote_code", False),
        field_messages=get_optional_str(dataset_cfg, "field_messages", "messages") or "messages",
        message_property_mappings=get_str_dict(
            dataset_cfg.get("message_property_mappings", {"role": "role", "content": "content"}),
            "dataset.message_property_mappings",
        ),
        refresh=get_optional_bool(dataset_cfg, "refresh", False),
        trace_roots=trace_roots,
        task_examples_dir=get_optional_str(dataset_cfg, "task_examples_dir"),
        sample_mode=get_optional_str(dataset_cfg, "sample_mode", "per_step") or "per_step",
        history_n=history_n,
        min_result=min_result,
        trace_source=trace_source,
        preflight_augmented_path=preflight_augmented_path,
        trace_root=trace_root,
        max_rollouts=max_rollouts,
        max_preflight_steps=max_preflight_steps,
        goal_variants=goal_variants,
        max_goal_variants=max_goal_variants,
        task_generation_root=task_generation_root,
        pretokenized_traces_dir=pretokenized_traces_dir,
        pretokenized_split=pretokenized_split,
        smoke_microactions=smoke_microactions,
        uitars=uitars,
    )

    target_modules_raw = lora_cfg.get("target_modules", LoraSettings().target_modules)
    if not isinstance(target_modules_raw, list) or not all(isinstance(m, str) for m in target_modules_raw):
        raise ValueError("`lora.target_modules` must be a list of strings.")

    lora = LoraSettings(
        r=get_optional_int(lora_cfg, "r", lora_cfg.get("lora_r", 16) if "lora_r" in lora_cfg else 16),
        alpha=get_optional_int(
            lora_cfg,
            "alpha",
            lora_cfg.get("lora_alpha", 32) if "lora_alpha" in lora_cfg else 32,
        ),
        target_modules=tuple(target_modules_raw),
        dropout=get_optional_float(lora_cfg, "dropout", 0.0),
    )

    max_steps = training_cfg.get("max_steps")
    if max_steps is not None and not isinstance(max_steps, int):
        raise ValueError("`training.max_steps` must be an integer.")

    save_lora_raw = training_cfg.get("save_lora_every_n_epochs", 1.0)
    if isinstance(save_lora_raw, bool) or not isinstance(save_lora_raw, (int, float)):
        raise ValueError("`training.save_lora_every_n_epochs` must be a number.")
    save_lora_every_n_epochs = float(save_lora_raw)

    eval_every_raw = training_cfg.get("eval_every_n_epochs", 0.5)
    if isinstance(eval_every_raw, bool) or not isinstance(eval_every_raw, (int, float)):
        raise ValueError("`training.eval_every_n_epochs` must be a number.")
    eval_every_n_epochs = float(eval_every_raw)

    mlflow_cfg = get_mapping(root, "mlflow")
    mlflow = MlflowSettings(
        enabled=get_optional_bool(mlflow_cfg, "enabled", False),
        experiment_name=get_optional_str(mlflow_cfg, "experiment_name", "dart-uitars-sft") or "dart-uitars-sft",
        run_name=get_optional_str(mlflow_cfg, "run_name"),
    )

    training = TrainingSettings(
        per_device_train_batch_size=get_optional_int(
            training_cfg,
            "per_device_train_batch_size",
            training_cfg.get("micro_batch_size", 1),
        ),
        gradient_accumulation_steps=get_optional_int(training_cfg, "gradient_accumulation_steps", 2),
        learning_rate=get_optional_float(training_cfg, "learning_rate", 2e-4),
        warmup_steps=get_optional_int(training_cfg, "warmup_steps", 10),
        logging_steps=get_optional_int(training_cfg, "logging_steps", 1),
        max_steps=max_steps,
        num_train_epochs=get_optional_int(training_cfg, "num_train_epochs", 1),
        save_steps=get_optional_int(training_cfg, "save_steps", 500),
        weight_decay=get_optional_float(training_cfg, "weight_decay", 0.0),
        packing=get_optional_bool(training_cfg, "packing", True),
        gradient_checkpointing=get_optional_bool(training_cfg, "gradient_checkpointing", True),
        fp16=get_optional_bool(training_cfg, "fp16", False),
        bf16=get_optional_bool(training_cfg, "bf16", True),
        ddp_find_unused_parameters=get_optional_bool(training_cfg, "ddp_find_unused_parameters", False),
        dataloader_num_workers=get_optional_nullable_int(training_cfg, "dataloader_num_workers"),
        dataloader_prefetch_factor=get_optional_nullable_int(training_cfg, "dataloader_prefetch_factor"),
        dataloader_persistent_workers=get_optional_bool(training_cfg, "dataloader_persistent_workers", False),
        save_lora_every_n_epochs=save_lora_every_n_epochs,
        eval_every_n_epochs=eval_every_n_epochs,
    )

    resolved_hub = HubSettings(
        push_to_hub=get_optional_bool(hub_cfg, "push_to_hub", True),
        model_id=get_optional_str(hub_cfg, "model_id"),
        private=get_optional_bool(hub_cfg, "private", True),
        revision=get_optional_str(hub_cfg, "revision"),
    )

    return ResolvedConfig(
        config_path=config_path,
        config_mode=config_mode,
        model=model,
        output_dir=repo_path(get_optional_str(output_cfg, "output_dir", "outputs/sft") or "outputs/sft"),
        datasets_dir=repo_path(get_optional_str(output_cfg, "datasets_dir", "datasets/runtime") or "datasets/runtime"),
        dataset=resolved_dataset,
        lora=lora,
        training=training,
        hf=HuggingFaceSettings(api_key=resolve_secret(get_optional_str(hf_cfg, "api_key"))),
        hub=resolved_hub,
        mlflow=mlflow,
    )


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_global_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def is_main_process() -> bool:
    return get_global_rank() == 0


class Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.streams[0], name)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def timestamped_output_enabled() -> bool:
    return env_flag("DART_SFT_TIMESTAMPED_OUTPUT") or bool(os.environ.get("DART_SFT_RUN_ID"))


def apply_timestamped_output_dir(config: ResolvedConfig) -> ResolvedConfig:
    if not timestamped_output_enabled():
        return config
    run_id = os.environ.get("DART_SFT_RUN_ID", "").strip()
    if not run_id:
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        os.environ["DART_SFT_RUN_ID"] = run_id
    run_root = os.environ.get("DART_SFT_RUN_ROOT", "").strip()
    output_root = repo_path(run_root) if run_root else config.output_dir
    return replace(config, output_dir=output_root / run_id)


def setup_rank_log(output_dir: Path) -> Path | None:
    if not env_flag("DART_SFT_SAVE_RANK_LOGS"):
        return None
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"rank-{get_global_rank():05d}.log"
    handle = open(log_path, "a", encoding="utf-8", buffering=1)
    _LOG_HANDLES.append(handle)
    sys.stdout = Tee(sys.stdout, handle)  # type: ignore[assignment]
    sys.stderr = Tee(sys.stderr, handle)  # type: ignore[assignment]
    return log_path


def write_run_metadata(config: ResolvedConfig) -> None:
    if not timestamped_output_enabled() or not is_main_process():
        return
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_id": os.environ.get("DART_SFT_RUN_ID", ""),
        "config_path": config.config_path,
        "output_dir": str(config.output_dir),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    with open(config.output_dir / "run_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def init_distributed() -> None:
    if not is_distributed():
        return
    import torch

    if torch.distributed.is_initialized():
        return
    local_rank = get_local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")


def distributed_barrier() -> None:
    if not is_distributed():
        return
    import torch

    if not torch.distributed.is_initialized():
        init_distributed()
    torch.distributed.barrier()


def wait_for_path(path: Path, *, timeout_s: float = 3600.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not path.is_file():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for staged dataset: {path}")
        time.sleep(1)


def configure_hf_env(config: ResolvedConfig) -> None:
    configure_hf_hub(config_token=config.hf.api_key)


def validate_hub_write(config: ResolvedConfig, *, skip: bool) -> None:
    if not config.hub.push_to_hub:
        return
    if skip:
        return
    if not config.hf.api_key:
        raise ValueError("`hf.api_key` is required when `hub.push_to_hub` is true.")
    if not config.hub.model_id:
        raise ValueError("`hub.model_id` is required when `hub.push_to_hub` is true.")
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Install `huggingface_hub` before running Hub-enabled training.") from exc
    api = HfApi(token=config.hf.api_key)
    api.whoami(token=config.hf.api_key)
    api.create_repo(
        repo_id=config.hub.model_id,
        token=config.hf.api_key,
        repo_type="model",
        private=config.hub.private,
        exist_ok=True,
    )


def trace_scan_settings_from_config(config: ResolvedConfig) -> TraceScanSettings:
    return TraceScanSettings(
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
        smoke_microactions=config.dataset.smoke_microactions,
    )


def resolve_staged_data_path(config: ResolvedConfig) -> Path:
    if config.dataset.format == "uitars_trace":
        scan_settings = trace_scan_settings_from_config(config)
        return config.datasets_dir / uitars_cache_name(scan_settings) / "data.jsonl"
    return config.datasets_dir / dataset_cache_name(config.dataset) / "data.jsonl"


def stage_dataset(config: ResolvedConfig, *, dry_run: bool = False) -> Path:
    if config.dataset.format == "uitars_trace":
        scan_settings = trace_scan_settings_from_config(config)
        data_path = resolve_staged_data_path(config)
        if dry_run:
            rows = build_trace_dataset_rows(scan_settings)
            if rows and (not data_path.is_file() or config.dataset.refresh):
                return stage_trace_dataset(scan_settings, config.datasets_dir, refresh=config.dataset.refresh)
            return data_path
        return stage_trace_dataset(scan_settings, config.datasets_dir, refresh=config.dataset.refresh)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install `datasets` before staging Hugging Face datasets.") from exc

    if not config.dataset.repo_id:
        raise ValueError("`dataset.repo_id` is required for chat datasets.")

    data_path = resolve_staged_data_path(config)
    dataset_dir = data_path.parent
    if data_path.is_file() and not config.dataset.refresh:
        return data_path

    dataset_dir.mkdir(parents=True, exist_ok=True)
    load_kwargs: dict[str, Any] = {
        "path": config.dataset.repo_id,
        "split": config.dataset.split,
        "trust_remote_code": config.dataset.trust_remote_code,
    }
    if config.dataset.name:
        load_kwargs["name"] = config.dataset.name
    if config.dataset.revision:
        load_kwargs["revision"] = config.dataset.revision
    if config.dataset.data_files is not None:
        load_kwargs["data_files"] = config.dataset.data_files
    if config.hf.api_key:
        load_kwargs["token"] = config.hf.api_key

    dataset = load_dataset(**load_kwargs)
    with open(data_path, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return data_path


def load_staged_dataset(data_path: Path):
    from datasets import load_dataset

    return load_dataset("json", data_files=str(data_path), split="train")


def normalize_messages(row: Mapping[str, Any], dataset: DatasetSettings) -> list[dict[str, str]]:
    raw = row.get(dataset.field_messages)
    if raw is None:
        raise ValueError(f"Missing field `{dataset.field_messages}` in dataset row.")
    if not isinstance(raw, list):
        raise ValueError(f"Field `{dataset.field_messages}` must be a list of message objects.")

    role_key = dataset.message_property_mappings.get("role", "role")
    content_key = dataset.message_property_mappings.get("content", "content")
    messages: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each message must be an object.")
        role = item.get(role_key)
        content = item.get(content_key)
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"Message must have string `{role_key}` and `{content_key}`.")
        messages.append({"role": role, "content": content})
    return messages


def prepare_train_dataset(data_path: Path, dataset_settings: DatasetSettings):
    dataset = load_staged_dataset(data_path)

    def add_messages(example: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
        return {"messages": normalize_messages(example, dataset_settings)}

    return dataset.map(add_messages, remove_columns=dataset.column_names)


def is_vision_model(model_name: str, trust_remote_code: bool, *, token: str | None = None) -> bool:
    from transformers import AutoConfig

    model_config = load_pretrained(
        AutoConfig,
        model_name,
        trust_remote_code=trust_remote_code,
        token=token,
    )
    model_type = getattr(model_config, "model_type", None)
    if isinstance(model_type, str) and model_type in VISION_MODEL_TYPES:
        return True
    architectures = getattr(model_config, "architectures", None) or []
    arch_text = " ".join(architectures).lower()
    return any(token in arch_text for token in ("vision", "vl", "image", "multimodal"))


def apply_eval_schedule(
    sft_config: Any,
    config: ResolvedConfig,
    *,
    train_dataset_len: int,
) -> None:
    if config.training.eval_every_n_epochs <= 0:
        sft_config.eval_strategy = "no"
        return
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    effective_batch = (
        config.training.per_device_train_batch_size
        * config.training.gradient_accumulation_steps
        * world_size
    )
    if config.training.max_steps is not None:
        steps_per_epoch = config.training.max_steps / max(float(config.training.num_train_epochs), 1.0)
    else:
        steps_per_epoch = train_dataset_len / max(effective_batch, 1)
    eval_steps = max(1, int(round(steps_per_epoch * config.training.eval_every_n_epochs)))
    sft_config.eval_strategy = "steps"
    sft_config.eval_steps = eval_steps
    sft_config.per_device_eval_batch_size = config.training.per_device_train_batch_size


def build_sft_config(config: ResolvedConfig, *, pretokenized: bool = False) -> Any:
    import torch
    from trl import SFTConfig

    use_fp16 = config.training.fp16 and torch.cuda.is_available()
    use_bf16 = config.training.bf16 and torch.cuda.is_available()
    if use_fp16 and use_bf16:
        use_fp16 = False
    packing = config.training.packing
    if config.dataset.format == "uitars_trace" or is_vision_model(
        config.model.base_model,
        config.model.trust_remote_code,
        token=configure_hf_hub(config_token=config.hf.api_key),
    ):
        packing = False
    configure_mlflow(config.mlflow)
    if config.training.save_lora_every_n_epochs > 0:
        save_strategy = "no"
    else:
        save_strategy = "steps"
    kwargs: dict[str, Any] = {
        "output_dir": str(config.output_dir),
        "per_device_train_batch_size": config.training.per_device_train_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "learning_rate": config.training.learning_rate,
        "warmup_steps": config.training.warmup_steps,
        "logging_steps": config.training.logging_steps,
        "weight_decay": config.training.weight_decay,
        "max_length": config.model.max_seq_length,
        "packing": packing,
        "fp16": use_fp16,
        "bf16": use_bf16,
        "gradient_checkpointing": config.training.gradient_checkpointing,
        "report_to": "none",
        "save_strategy": save_strategy,
        "eval_strategy": "no",
        "remove_unused_columns": False,
    }
    if save_strategy == "steps":
        kwargs["save_steps"] = config.training.save_steps
    if config.training.max_steps is not None:
        kwargs["max_steps"] = config.training.max_steps
    else:
        kwargs["num_train_epochs"] = config.training.num_train_epochs
    if config.dataset.format == "uitars_trace":
        kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}
    if config.hub.push_to_hub:
        kwargs["push_to_hub"] = True
        kwargs["hub_model_id"] = config.hub.model_id
        if config.hub.revision:
            kwargs["hub_revision"] = config.hub.revision
    if pretokenized:
        kwargs["accelerator_config"] = {"dispatch_batches": False}
        num_workers = 2 if config.training.dataloader_num_workers is None else config.training.dataloader_num_workers
        kwargs["dataloader_num_workers"] = num_workers
        if config.training.dataloader_prefetch_factor is not None:
            kwargs["dataloader_prefetch_factor"] = config.training.dataloader_prefetch_factor
        elif num_workers > 0:
            kwargs["dataloader_prefetch_factor"] = 2
        kwargs["dataloader_persistent_workers"] = (
            config.training.dataloader_persistent_workers or num_workers > 0
        )
    kwargs.update(distributed_sft_overrides(config))
    return SFTConfig(**kwargs)


def distributed_sft_overrides(config: ResolvedConfig) -> dict[str, Any]:
    if not is_distributed():
        return {}
    return {
        "ddp_find_unused_parameters": config.training.ddp_find_unused_parameters,
    }


def load_model_and_processor(config: ResolvedConfig):
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    trust_remote_code = config.model.trust_remote_code
    hf_token = configure_hf_hub(config_token=config.hf.api_key)
    use_bf16 = config.training.bf16 and torch.cuda.is_available()
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_vision = config.dataset.format == "uitars_trace" or is_vision_model(
        config.model.base_model,
        trust_remote_code,
        token=hf_token,
    )

    if use_vision:
        processor = load_pretrained(
            AutoProcessor,
            config.model.base_model,
            trust_remote_code=trust_remote_code,
            token=hf_token,
        )
        tokenizer = processor.tokenizer
    else:
        processor = None
        tokenizer = load_pretrained(
            AutoTokenizer,
            config.model.base_model,
            trust_remote_code=trust_remote_code,
            token=hf_token,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "torch_dtype": model_dtype,
    }
    if is_distributed():
        local_rank = get_local_rank()
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    else:
        model_kwargs["device_map"] = "auto"

    if use_vision:
        model = load_pretrained(
            AutoModelForImageTextToText,
            config.model.base_model,
            trust_remote_code=trust_remote_code,
            token=hf_token,
            **model_kwargs,
        )
    else:
        model = load_pretrained(
            AutoModelForCausalLM,
            config.model.base_model,
            trust_remote_code=trust_remote_code,
            token=hf_token,
            **model_kwargs,
        )

    peft_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=list(config.lora.target_modules),
        task_type="CAUSAL_LM",
    )
    return model, processor, tokenizer, peft_config


def print_dry_run_summary(
    config: ResolvedConfig,
    data_path: Path | None,
    sft_config: Any,
    *,
    pretokenized_dir: Path | None = None,
) -> None:
    if pretokenized_dir is not None:
        print(f"Pretokenized dataset: {pretokenized_dir}")
    else:
        print(f"Staged dataset: {data_path}")
    print(f"Dataset format: {config.dataset.format}")
    print(f"Output dir: {config.output_dir}")
    print(f"Base model: {config.model.base_model}")
    print(f"Max seq length: {config.model.max_seq_length}")
    dtype_label = "bfloat16" if config.training.bf16 else "float16"
    print(f"Model dtype: {dtype_label}")
    print(f"LoRA r/alpha: {config.lora.r}/{config.lora.alpha}")
    print(f"MLflow enabled: {config.mlflow.enabled}")
    print(f"LoRA checkpoint every n epochs: {config.training.save_lora_every_n_epochs}")
    print(f"Eval every n epochs: {config.training.eval_every_n_epochs}")
    print(f"Training: {sft_config}")


def prepare_staged_dataset(
    config: ResolvedConfig,
    *,
    dry_run: bool = False,
) -> Path:
    data_path = resolve_staged_data_path(config)
    if is_main_process():
        data_path = stage_dataset(config, dry_run=dry_run)
    if is_distributed():
        distributed_barrier()
        if not is_main_process():
            wait_for_path(data_path)
    return data_path


def ensure_pretokenized_split(config: ResolvedConfig, pretokenized_dir: Path) -> None:
    from microaction_split import compute_split_for_training_rows, load_split, write_split
    from pretokenized_manifest import load_jsonl_rows

    if not config.dataset.task_examples_dir:
        raise ValueError(
            "dataset.task_examples_dir is required to compute pretokenized train/val split metadata."
        )
    staged_jsonl = resolve_staged_data_path(config)
    rows = load_jsonl_rows(staged_jsonl)
    split = compute_split_for_training_rows(
        rows,
        task_examples_dir=config.dataset.task_examples_dir,
        task_generation_root=config.dataset.task_generation_root,
        goal_variants=config.dataset.goal_variants,
        max_goal_variants=config.dataset.max_goal_variants,
        smoke_microactions=config.dataset.smoke_microactions,
    )
    existing = load_split(pretokenized_dir)
    if existing is not None and existing.to_dict() == split.to_dict():
        return
    write_split(pretokenized_dir, split)


def train(
    config: ResolvedConfig,
    data_path: Path | None = None,
    *,
    pretokenized_dir: Path | None = None,
) -> None:
    from trl import SFTTrainer

    if is_distributed():
        init_distributed()
    if is_main_process():
        config.output_dir.mkdir(parents=True, exist_ok=True)
    if is_distributed():
        distributed_barrier()
    sft_config = build_sft_config(config, pretokenized=pretokenized_dir is not None)
    model, processor, tokenizer, peft_config = load_model_and_processor(config)

    if pretokenized_dir is not None:
        if is_main_process():
            ensure_pretokenized_split(config, pretokenized_dir)
        if is_distributed():
            distributed_barrier()
        train_dataset = PretokenizedUitarsDataset(
            pretokenized_dir,
            split=config.dataset.pretokenized_split,
        )
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = processor.tokenizer.eos_token_id
        data_collator = make_pretokenized_collator(pad_token_id)
        processing_class = processor
    elif config.dataset.format == "uitars_trace":
        if data_path is None:
            raise ValueError("data_path is required for uitars_trace training.")
        train_dataset = UitarsTraceDataset(data_path, config.dataset.uitars)
        data_collator = make_uitars_data_collator(processor)
        processing_class = processor
    else:
        train_dataset = prepare_train_dataset(data_path, config.dataset)
        data_collator = None
        processing_class = tokenizer

    eval_dataset = None
    if (
        pretokenized_dir is not None
        and config.training.eval_every_n_epochs > 0
        and config.dataset.pretokenized_split == "train"
    ):
        eval_dataset = PretokenizedUitarsDataset(pretokenized_dir, split="val")
        if len(eval_dataset) == 0:
            eval_dataset = None

    if eval_dataset is not None:
        apply_eval_schedule(sft_config, config, train_dataset_len=len(train_dataset))

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_config,
        "train_dataset": train_dataset,
        "processing_class": processing_class,
        "peft_config": peft_config,
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    if data_collator is not None:
        trainer_kwargs["data_collator"] = data_collator

    callbacks: list[Any] = []
    if config.mlflow.enabled and configure_mlflow(config.mlflow):
        callbacks.append(build_mlflow_callback())
    if config.training.save_lora_every_n_epochs > 0:
        callbacks.append(
            LoraEpochCheckpointCallback(
                every_n_epochs=config.training.save_lora_every_n_epochs,
                output_dir=config.output_dir,
            )
        )
    if callbacks:
        trainer_kwargs["callbacks"] = callbacks

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    if is_main_process():
        trainer.save_model(str(config.output_dir))
        if config.hub.push_to_hub:
            trainer.push_to_hub()


def repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def dataset_cache_name(dataset: DatasetSettings) -> str:
    if dataset.format == "uitars_trace":
        scan_settings = TraceScanSettings(
            trace_roots=dataset.trace_roots,
            task_examples_dir=dataset.task_examples_dir or "",
            sample_mode=dataset.sample_mode,
            history_n=dataset.history_n,
            min_result=dataset.min_result,
            trace_source=dataset.trace_source,  # type: ignore[arg-type]
            uitars=dataset.uitars,
            preflight_augmented_path=dataset.preflight_augmented_path,
            trace_root=dataset.trace_root,
            max_rollouts=dataset.max_rollouts,
            max_preflight_steps=dataset.max_preflight_steps,
            goal_variants=dataset.goal_variants,
            max_goal_variants=dataset.max_goal_variants,
            task_generation_root=dataset.task_generation_root,
            pretokenized_traces_dir=dataset.pretokenized_traces_dir,
            smoke_microactions=dataset.smoke_microactions,
        )
        return uitars_cache_name(scan_settings)
    if not dataset.repo_id:
        raise ValueError("`dataset.repo_id` is required for chat datasets.")
    raw = json.dumps(
        {
            "repo_id": dataset.repo_id,
            "split": dataset.split,
            "name": dataset.name,
            "revision": dataset.revision,
            "data_files": dataset.data_files,
        },
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    safe_repo = "".join(char if char.isalnum() or char in "._-" else "-" for char in dataset.repo_id)
    return f"{safe_repo}-{digest}"


def resolve_secret(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    if value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1])
    if value.startswith("$") and len(value) > 1:
        return os.environ.get(value[1:])
    return value


def get_mapping(root: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = root.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"`{key}` must be a YAML object.")
    return dict(value)


def get_required_str(root: Mapping[str, Any], key: str, context: str) -> str:
    value = root.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{context}.{key}` must be a non-empty string.")
    return value.strip()


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


def get_optional_nullable_int(root: Mapping[str, Any], key: str) -> int | None:
    value = root.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"`{key}` must be an integer.")
    return value


def get_optional_float(root: Mapping[str, Any], key: str, default: float) -> float:
    value = root.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"`{key}` must be a number.")
    return float(value)


def get_optional_bool(root: Mapping[str, Any], key: str, default: bool) -> bool:
    value = root.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"`{key}` must be a boolean.")
    return value


def get_str_dict(value: Any, context: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"`{context}` must be a YAML object.")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ValueError(f"`{context}` must contain string keys and values.")
        result[key] = item
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    if args.skip_hf_validation and not args.dry_run:
        parser.error("`--skip-hf-validation` is only allowed with `--dry-run`.")
    reject_mixed_config_usage(parser, raw_argv, args)
    configure_hf_hub()
    config = load_config_file(args.config) if args.config else resolve_args(args)
    config = apply_timestamped_output_dir(config)
    setup_rank_log(config.output_dir)
    write_run_metadata(config)
    configure_hf_env(config)
    if is_main_process():
        validate_hub_write(config, skip=args.skip_hf_validation)
    pretokenized_dir: Path | None = None
    if args.from_pretokenized_traces:
        pretokenized_dir = repo_path(args.from_pretokenized_traces)
    elif config.dataset.pretokenized_traces_dir:
        pretokenized_dir = repo_path(config.dataset.pretokenized_traces_dir)
    if pretokenized_dir is not None:
        if args.dry_run:
            if is_main_process():
                print_dry_run_summary(
                    config,
                    None,
                    build_sft_config(config, pretokenized=True),
                    pretokenized_dir=pretokenized_dir,
                )
            return 0
        train(config, pretokenized_dir=pretokenized_dir)
        return 0
    data_path = prepare_staged_dataset(config, dry_run=args.dry_run)
    if args.dry_run:
        if is_main_process():
            print_dry_run_summary(config, data_path, build_sft_config(config))
        return 0
    train(config, data_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
