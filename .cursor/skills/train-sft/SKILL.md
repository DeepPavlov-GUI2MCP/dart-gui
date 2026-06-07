---
name: train-sft
description: Train FP16 LoRA adapters for Qwen-like or UI-TARS models with TRL. Use when preparing SFT datasets, editing `training/run_sft.py` configs, launching TRL SFT training, or publishing adapters to Hugging Face Hub.
---

# Train SFT (TRL LoRA)

Use `training/run_sft.py` to stage an SFT dataset, validate Hub write access, and launch TRL `SFTTrainer` with FP16 LoRA.

## Setup

```bash
cd /home/pitchblack/dart-gui
source .venv/bin/activate
uv pip install trl peft datasets huggingface-hub pyyaml
```

GPU training requires a CUDA-capable environment with the Torch stack from [requirements.txt](../../requirements.txt).

## Dataset Formats

### UI-TARS writer traces (recommended for GUI SFT)

Set `train_sft.dataset.format: uitars_trace` and point at eval rollout directories containing `traj.jsonl`, screenshots, and `result.txt`.

```yaml
dataset:
  format: uitars_trace
  trace_roots:
    - GUI-Docker-Env/results_uitars15_writer_full_20260530_143529/pyautogui/screenshot/ui_tars_1.5
  task_examples_dir: GUI-Docker-Env/evaluation_examples/examples
  sample_mode: per_step          # default; one row per step, loss on final assistant only
  history_n: 5
  min_result: 1.0                # optional
  uitars:
    prompt_style: qwen25vl_normal
    infer_mode: qwen25vl_normal
    language: English
    max_pixels: 12845056
    min_pixels: 78400
```

Build or inspect staged JSONL without training:

```bash
python training/build_uitars_sft_dataset.py --config training/configs/sft_example.yml --dry-run
```

Use `sample_mode: full_trajectory` to emit one row per rollout and train all assistant turns.

Rollout dirs must include an initial observation screenshot (`initial.png`, `initial_screenshot.png`, or `step_0*.png`) so step 1 can be reconstructed.

### Legacy chat JSONL

Set `train_sft.dataset.format: chat` (default when omitted) and provide a Hugging Face dataset repo with OpenAI-style `messages`:

```json
{"messages":[{"role":"user","content":"Task instruction"},{"role":"assistant","content":"Response"}]}
```

See `datasets/templates/sft_messages.jsonl`. For FineTome-style data, use `training/configs/sft_example_chat.yml`.

## Config

Start from `training/configs/sft_example.yml` for UI-TARS trace SFT, or `training/configs/sft_example_chat.yml` for legacy chat smoke tests.

Required fields for trace SFT:

- `train_sft.model.base_model`: for example `ByteDance-Seed/UI-TARS-1.5-7B`
- `train_sft.dataset.trace_roots`: one or more rollout root directories
- `train_sft.dataset.task_examples_dir`: OSWorld task JSON examples root

Hub upload is enabled in the example config. Disable it explicitly:

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
python training/run_sft.py --config training/configs/sft_example.yml --dry-run --skip-hf-validation
```

Use `--skip-hf-validation` only for dry-run debugging when no Hub repo should be created.

## Outputs

- Runtime datasets: `datasets/runtime/` (ignored by git).
- Model outputs: configured by `train_sft.output.output_dir`.
