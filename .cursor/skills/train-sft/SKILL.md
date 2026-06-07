---
name: train-sft
description: Train QLoRA adapters for Qwen-like or UI-TARS models with TRL. Use when preparing SFT datasets, editing `training/run_sft.py` configs, launching TRL SFT training, or publishing adapters to Hugging Face Hub.
---

# Train SFT (TRL QLoRA)

Use `training/run_sft.py` to stage an SFT dataset from Hugging Face, validate Hub write access, and launch TRL `SFTTrainer` with 4-bit QLoRA.

## Setup

```bash
cd /home/pitchblack/dart-gui
source .venv/bin/activate
uv pip install trl bitsandbytes datasets huggingface-hub pyyaml
```

GPU training requires a CUDA-capable environment with the Torch stack from [requirements.txt](../../requirements.txt).

## Dataset Format

Use OpenAI-style SFT records:

```json
{"messages":[{"role":"user","content":"Task instruction"},{"role":"assistant","content":"Response"}]}
```

The tracked template is `datasets/templates/sft_messages.jsonl`. Real runs should upload the same format to a Hugging Face dataset repo and set `train_sft.dataset.repo_id`.

For FineTome-style data, map `conversations` with:

```yaml
field_messages: conversations
message_property_mappings:
  role: from
  content: value
```

## Config

Start from `training/configs/sft_example.yml` and copy it for local use.

Required fields:

- `train_sft.model.base_model`: Qwen-like base model, for example `ByteDance-Seed/UI-TARS-1.5-7B`.
- `train_sft.dataset.repo_id`: Hugging Face dataset repo loaded at runtime.
- `train_sft.hf.api_key`: token value or `${HUGGINGFACE_API_KEY}`.
- `train_sft.hub.model_id`: target model repo when `hub.push_to_hub` is true.

Hub upload is enabled by default. Disable it explicitly:

```yaml
hub:
  push_to_hub: false
```

## Run

```bash
cd /home/pitchblack/dart-gui
source .venv/bin/activate
python training/run_sft.py --config training/configs/sft_example.yml
```

Dry-run without loading model or training:

```bash
python training/run_sft.py --config training/configs/sft_example.yml --dry-run
```

Use `--skip-hf-validation` only for dry-run debugging when no Hub repo should be created.

## Outputs

- Runtime datasets: `datasets/runtime/` (ignored by git).
- Model outputs: configured by `train_sft.output.output_dir`.
