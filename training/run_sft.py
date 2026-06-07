from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

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
    repo_id: str
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


@dataclass(frozen=True)
class ModelSettings:
    base_model: str
    max_seq_length: int = 2048
    load_in_4bit: bool = True
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
    bf16: bool = True


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train models with TRL SFT (QLoRA).")
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
        load_in_4bit=get_optional_bool(model_cfg, "load_in_4bit", True),
        trust_remote_code=get_optional_bool(model_cfg, "trust_remote_code", True),
    )

    resolved_dataset = DatasetSettings(
        repo_id=get_required_str(dataset_cfg, "repo_id", "dataset"),
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
        bf16=get_optional_bool(training_cfg, "bf16", True),
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
    )


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


def is_vision_model(model_name: str, trust_remote_code: bool) -> bool:
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model_type = getattr(model_config, "model_type", None)
    if isinstance(model_type, str) and model_type in VISION_MODEL_TYPES:
        return True
    architectures = getattr(model_config, "architectures", None) or []
    arch_text = " ".join(architectures).lower()
    return any(token in arch_text for token in ("vision", "vl", "image", "multimodal"))


def build_sft_config(config: ResolvedConfig) -> Any:
    import torch
    from trl import SFTConfig

    use_bf16 = config.training.bf16 and torch.cuda.is_available()
    kwargs: dict[str, Any] = {
        "output_dir": str(config.output_dir),
        "per_device_train_batch_size": config.training.per_device_train_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "learning_rate": config.training.learning_rate,
        "warmup_steps": config.training.warmup_steps,
        "logging_steps": config.training.logging_steps,
        "save_steps": config.training.save_steps,
        "weight_decay": config.training.weight_decay,
        "max_seq_length": config.model.max_seq_length,
        "packing": config.training.packing,
        "bf16": use_bf16,
        "gradient_checkpointing": config.training.gradient_checkpointing,
        "report_to": "none",
        "remove_unused_columns": False,
    }
    if config.training.max_steps is not None:
        kwargs["max_steps"] = config.training.max_steps
    else:
        kwargs["num_train_epochs"] = config.training.num_train_epochs
    if config.hub.push_to_hub:
        kwargs["push_to_hub"] = True
        kwargs["hub_model_id"] = config.hub.model_id
        if config.hub.revision:
            kwargs["hub_revision"] = config.hub.revision
    return SFTConfig(**kwargs)


def load_model_and_tokenizer(config: ResolvedConfig):
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

    trust_remote_code = config.model.trust_remote_code
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.base_model,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "device_map": "auto",
    }
    if config.model.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    if is_vision_model(config.model.base_model, trust_remote_code):
        model = AutoModelForImageTextToText.from_pretrained(config.model.base_model, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(config.model.base_model, **model_kwargs)

    peft_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=list(config.lora.target_modules),
        task_type="CAUSAL_LM",
    )
    return model, tokenizer, peft_config


def print_dry_run_summary(config: ResolvedConfig, data_path: Path, sft_config: Any) -> None:
    print(f"Staged dataset: {data_path}")
    print(f"Output dir: {config.output_dir}")
    print(f"Base model: {config.model.base_model}")
    print(f"Max seq length: {config.model.max_seq_length}")
    print(f"Load in 4bit: {config.model.load_in_4bit}")
    print(f"LoRA r/alpha: {config.lora.r}/{config.lora.alpha}")
    print(f"Training: {sft_config}")


def train(config: ResolvedConfig, data_path: Path) -> None:
    from trl import SFTTrainer

    config.output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = prepare_train_dataset(data_path, config.dataset)
    sft_config = build_sft_config(config)
    model, tokenizer, peft_config = load_model_and_tokenizer(config)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(config.output_dir))
    if config.hub.push_to_hub:
        trainer.push_to_hub()


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
    config = load_config_file(args.config) if args.config else resolve_args(args)
    configure_hf_env(config)
    validate_hub_write(config, skip=args.skip_hf_validation)
    data_path = stage_dataset(config)
    sft_config = build_sft_config(config)
    if args.dry_run:
        print_dry_run_summary(config, data_path, sft_config)
        return 0
    train(config, data_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
