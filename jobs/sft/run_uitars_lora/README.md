# UI-TARS LoRA SFT job

ML Space `pytorch2` job for training UI-TARS LoRA adapters from pretokenized trace
shards via [`training/run_sft.py`](../../../training/run_sft.py).

## Base image

Primary: `cr.ai.cloud.ru/aicloud-base-images/py3.11-torch2.9.0:0.0.42`

Fallback if unavailable in your region: `py3.11-torch2.7.0:0.0.42`

The image inherits bundled torch/CUDA from the base image. Job
[`requirements.txt`](requirements.txt) is derived from
[`dart_rollouter/requirements_cuda_129_holotron.txt`](../../../dart_rollouter/requirements_cuda_129_holotron.txt)
with `torch`, CUDA wheels, `vllm`, and inference-only packages removed to avoid
multi-GB reinstalls and ABI conflicts.

## Build

From repo root:

```bash
export IMAGE_NAME="job-dart-uitars-sft:1.0"
bash jobs/sft/run_uitars_lora/build.sh
```

## Presubmit

Runs automatically before submit. Validates that:

- `DART_SFT_CONFIG` points to a `uitars_trace` config with `pretokenized_traces_dir`
- Staged `data.jsonl` exists
- Pretokenized `shard-*.pt` files exist and load correctly
- Training row count matches staged rows (or goal-variant expansion)
- `manifest.json` metadata matches when present

Manual run:

```bash
export DART_SFT_CONFIG="training/configs/sft_holo_goal_variants_fixture.yml"
python jobs/sft/run_uitars_lora/presubmit.py
```

## Worker environment

| Variable | Required | Purpose |
|----------|----------|---------|
| `DART_SFT_CONFIG` | yes | Path to `train_sft` YAML |
| `DART_SFT_PRETOKENIZED_DIR` | no | Override `dataset.pretokenized_traces_dir` |
| `DART_JOB_GPUS_PER_NODE` | no | GPUs per node (default 1) |
| `DART_JOB_NUM_NODES` | no | Worker nodes (default 1) |
| `HF_HOME` / `HF_TOKEN` | recommended | Base model cache on NFS |
| `DART_SFT_PYTHON` | no | Default `/opt/dart-sft-venv/bin/python3` |

Effective batch size =
`per_device_train_batch_size * DART_JOB_GPUS_PER_NODE * DART_JOB_NUM_NODES * gradient_accumulation_steps`.
