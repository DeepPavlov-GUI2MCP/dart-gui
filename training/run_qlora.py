from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetSettings:
    repo_id: str
    split: str = "train"
    name: str | None = None
    revision: str | None = None
    data_files: Any = None
    trust_remote_code: bool = False
    type: str = "chat_template"
    field_messages: str = "messages"
    message_property_mappings: dict[str, str] = field(default_factory=lambda: {"role": "role", "content": "content"})
    refresh: bool = False


@dataclass(frozen=True)
class HuggingFaceSettings:
    api_key: str | None = None


@dataclass(frozen=True)
class HubSettings:
    push_to_hub: bool = True
    model_id: str | None = None
    private: bool = True
    strategy: str | None = "checkpoint"
    revision: str | None = None


@dataclass(frozen=True)
class ResolvedConfig:
    config_path: str | None
    config_mode: bool
    base_model: str
    chat_template: str
    sequence_len: int
    output_dir: Path
    datasets_dir: Path
    dataset_prepared_path: Path
    generated_config_dir: Path
    dataset: DatasetSettings
    hf: HuggingFaceSettings
    hub: HubSettings
    axolotl_command: tuple[str, ...]
    axolotl_overrides: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Qwen-like models with Axolotl QLoRA.")
    parser.add_argument("--config", type=str, default=None, help="Path to a full YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Generate config and stage data without launching Axolotl.")
    parser.add_argument("--skip-hf-validation", action="store_true", help="Skip Hub write validation for dry-run debugging.")

    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--dataset_repo", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hf_api_key", type=str, default=None)
    parser.add_argument("--no_push_to_hub", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="outputs/qlora")
    return parser


def reject_mixed_config_usage(parser: argparse.ArgumentParser, argv: Sequence[str], args: argparse.Namespace) -> None:
    if not args.config:
        return
    allowed = {"--config", "--dry-run", "--skip-hf-validation"}
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
    root = payload.get("train_qlora", payload)
    if not isinstance(root, dict):
        raise ValueError(f"`train_qlora` config must be a YAML object: {path}")
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
    model = get_mapping(root, "model")
    dataset = get_mapping(root, "dataset")
    output = get_mapping(root, "output")
    hf = get_mapping(root, "hf")
    hub = get_mapping(root, "hub")
    axolotl = get_mapping(root, "axolotl")
    lora = get_mapping(root, "lora")
    training = get_mapping(root, "training")

    base_model = get_required_str(model, "base_model", "model")
    datasets_dir = repo_path(get_optional_str(output, "datasets_dir", "datasets/runtime"))
    output_dir = repo_path(get_optional_str(output, "output_dir", "outputs/qlora"))
    generated_config_dir = repo_path(get_optional_str(output, "generated_config_dir", "outputs/qlora/generated_configs"))
    dataset_prepared_path = repo_path(get_optional_str(output, "dataset_prepared_path", "outputs/qlora/prepared"))

    resolved_dataset = DatasetSettings(
        repo_id=get_required_str(dataset, "repo_id", "dataset"),
        split=get_optional_str(dataset, "split", "train"),
        name=get_optional_str(dataset, "name"),
        revision=get_optional_str(dataset, "revision"),
        data_files=dataset.get("data_files"),
        trust_remote_code=get_optional_bool(dataset, "trust_remote_code", False),
        type=get_optional_str(dataset, "type", "chat_template"),
        field_messages=get_optional_str(dataset, "field_messages", "messages"),
        message_property_mappings=get_str_dict(
            dataset.get("message_property_mappings", {"role": "role", "content": "content"}),
            "dataset.message_property_mappings",
        ),
        refresh=get_optional_bool(dataset, "refresh", False),
    )

    resolved_hub = HubSettings(
        push_to_hub=get_optional_bool(hub, "push_to_hub", True),
        model_id=get_optional_str(hub, "model_id"),
        private=get_optional_bool(hub, "private", True),
        strategy=get_optional_str(hub, "strategy", "checkpoint"),
        revision=get_optional_str(hub, "revision"),
    )

    axolotl_overrides = default_axolotl_config()
    axolotl_overrides.update(get_mapping(root, "overrides"))
    axolotl_overrides.update(training)
    axolotl_overrides.update(lora)

    command_value = axolotl.get("command", ["axolotl", "train"])
    command = parse_command(command_value)

    return ResolvedConfig(
        config_path=config_path,
        config_mode=config_mode,
        base_model=base_model,
        chat_template=get_optional_str(model, "chat_template", "qwen3"),
        sequence_len=get_optional_int(model, "sequence_len", 2048),
        output_dir=output_dir,
        datasets_dir=datasets_dir,
        dataset_prepared_path=dataset_prepared_path,
        generated_config_dir=generated_config_dir,
        dataset=resolved_dataset,
        hf=HuggingFaceSettings(api_key=resolve_secret(get_optional_str(hf, "api_key"))),
        hub=resolved_hub,
        axolotl_command=command,
        axolotl_overrides=axolotl_overrides,
    )


def default_axolotl_config() -> dict[str, Any]:
    return {
        "strict": False,
        "val_set_size": 0.0,
        "sample_packing": True,
        "eval_sample_packing": True,
        "pad_to_sequence_len": True,
        "load_in_4bit": True,
        "adapter": "qlora",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj"],
        "gradient_accumulation_steps": 2,
        "micro_batch_size": 1,
        "num_epochs": 1,
        "optimizer": "adamw_torch_4bit",
        "lr_scheduler": "cosine",
        "learning_rate": 0.0002,
        "bf16": "auto",
        "tf32": True,
        "gradient_checkpointing": "offload",
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_steps": 1,
        "flash_attention": True,
        "warmup_steps": 10,
        "evals_per_epoch": 0,
        "saves_per_epoch": 1,
        "weight_decay": 0.0,
    }


def configure_hf_env(config: ResolvedConfig) -> None:
    if not config.hf.api_key:
        return
    for name in ("HF_API_KEY", "HUGGINGFACE_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        os.environ[name] = config.hf.api_key


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


def stage_dataset(config: ResolvedConfig) -> Path:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install `datasets` before staging Hugging Face datasets.") from exc

    dataset_dir = config.datasets_dir / dataset_cache_name(config.dataset)
    data_path = dataset_dir / "data.jsonl"
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


def write_axolotl_config(config: ResolvedConfig, dataset_path: Path) -> Path:
    config.generated_config_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.dataset_prepared_path.mkdir(parents=True, exist_ok=True)

    payload = dict(config.axolotl_overrides)
    payload.update(
        {
            "base_model": config.base_model,
            "chat_template": config.chat_template,
            "sequence_len": config.sequence_len,
            "output_dir": str(config.output_dir),
            "dataset_prepared_path": str(config.dataset_prepared_path),
            "datasets": [
                {
                    "path": str(dataset_path),
                    "ds_type": "json",
                    "type": config.dataset.type,
                    "field_messages": config.dataset.field_messages,
                    "message_property_mappings": config.dataset.message_property_mappings,
                }
            ],
        }
    )
    if config.hf.api_key:
        payload["hf_use_auth_token"] = True
    if config.hub.push_to_hub:
        payload["hub_model_id"] = config.hub.model_id
        payload["hub_strategy"] = config.hub.strategy
        if config.hub.revision:
            payload["hub_revision"] = config.hub.revision
    payload = {key: value for key, value in payload.items() if value is not None}

    config_path = config.generated_config_dir / "axolotl_qlora.yml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)
    return config_path


def run_axolotl(config: ResolvedConfig, axolotl_config_path: Path, *, dry_run: bool) -> None:
    if dry_run:
        print(f"Generated Axolotl config: {axolotl_config_path}")
        return
    subprocess.run([*config.axolotl_command, str(axolotl_config_path)], cwd=REPO_ROOT, check=True)


def repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def dataset_cache_name(dataset: DatasetSettings) -> str:
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


def parse_command(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        parts = tuple(shlex.split(value))
    elif isinstance(value, list) and all(isinstance(item, str) and item for item in value):
        parts = tuple(value)
    else:
        raise ValueError("`axolotl.command` must be a string or list of strings.")
    if not parts:
        raise ValueError("`axolotl.command` cannot be empty.")
    return parts


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
    config = load_config_file(args.config) if args.config else resolve_args(args)
    configure_hf_env(config)
    validate_hub_write(config, skip=args.skip_hf_validation)
    dataset_path = stage_dataset(config)
    axolotl_config_path = write_axolotl_config(config, dataset_path)
    run_axolotl(config, axolotl_config_path, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
