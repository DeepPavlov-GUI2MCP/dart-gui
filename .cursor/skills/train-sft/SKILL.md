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

Each `traj.jsonl` row's `screenshot_file` is the pre-action observation for that step; step 1's screenshot is the initial desktop state.

Set `trace_source: uitars` to force native UI-TARS traces. Default is `auto`.

### Holo writer traces

Holo rollouts (for example `GUI-Docker-Env/results_holo_gpt54_writer_traces_*`) use a different artifact timeline:

- Step 1: initial screenshot (`response: null`)
- Preflight rows: a11y setup steps (`response: null`, not trained)
- Agent rows: Holo Step JSON in `response`, screenshot captured **after** the action

Use:

```yaml
dataset:
  format: uitars_trace
  trace_source: holo          # or auto (detects preflight rows / Holo JSON)
  trace_roots:
    - GUI-Docker-Env/results_holo_gpt54_writer_traces_20260607_031517/pyautogui/screenshot/Hcompany/Holotron-3-Nano
  task_examples_dir: GUI-Docker-Env/evaluation_examples/examples
  sample_mode: per_step
```

Holo JSON responses are converted to UI-TARS `Thought:` / `Action:` labels (`fail` → `call_user()`). Preflight screenshots are used only to locate the first agent pre-action frame.

Omit `min_result` for trace-collection runs where `result.txt` is still `-1`.

Example config: `training/configs/sft_holo_example.yml`.

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
