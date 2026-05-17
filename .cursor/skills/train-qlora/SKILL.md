---
name: train-qlora
description: Train QLoRA adapters for Qwen-like or UI-TARS models with Axolotl. Use when preparing SFT datasets, editing `training/run_qlora.py` configs, launching Axolotl QLoRA training, or publishing adapters to Hugging Face Hub.
---

# Train QLoRA

Use `training/run_qlora.py` to stage an SFT dataset from Hugging Face, generate an Axolotl config, validate Hub write access, and launch QLoRA training.

## Setup

```bash
cd /home/pitchblack/dart-gui
source .venv/bin/activate
uv pip install -e axolotl
uv pip install datasets huggingface-hub pyyaml
```

For Axolotl GPU dependencies, follow the Axolotl submodule docs if the current `.venv` does not already include the required Torch/CUDA stack.

## Dataset Format

Use OpenAI-style SFT records:

```json
{"messages":[{"role":"user","content":"Task instruction"},{"role":"assistant","content":"Response"}]}
```

The tracked template is `datasets/templates/sft_messages.jsonl`. Real runs should upload the same format to a Hugging Face dataset repo and set `train_qlora.dataset.repo_id`.

For FineTome-style data, map `conversations` with:

```yaml
field_messages: conversations
message_property_mappings:
  role: from
  content: value
```

## Config

Start from `training/configs/qlora_example.yml` and copy it for local use.

Required fields:

- `train_qlora.model.base_model`: Qwen-like base model, for example `ByteDance-Seed/UI-TARS-1.5-7B`.
- `train_qlora.dataset.repo_id`: Hugging Face dataset repo loaded at runtime.
- `train_qlora.hf.api_key`: token value or `${HUGGINGFACE_API_KEY}`.
- `train_qlora.hub.model_id`: target model repo when `hub.push_to_hub` is true.

Hub upload is enabled by default. Disable it explicitly:

```yaml
hub:
  push_to_hub: false
```

## Run

```bash
cd /home/pitchblack/dart-gui
source .venv/bin/activate
python training/run_qlora.py --config training/configs/qlora_example.yml
```

Dry-run without launching training:

```bash
python training/run_qlora.py --config training/configs/qlora_example.yml --dry-run
```

Use `--skip-hf-validation` only for dry-run debugging when no Hub repo should be created.

## Outputs

- Runtime datasets: `datasets/runtime/` (ignored by git).
- Generated Axolotl config: `outputs/qlora/generated_configs/axolotl_qlora.yml`.
- Model outputs: configured by `train_qlora.output.output_dir`.
