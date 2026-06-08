# ML Space jobs

Batch jobs for ML Space (Cloud.ru): Docker images, worker entry scripts, and a submit CLI
backed by `client_lib`.

## Repository layout

```
jobs/
  submit.py
  hf_cache.py
  README.md
  sft/run_uitars_lora/
    Dockerfile
    requirements.txt
    build.sh
    run.py
    presubmit.py
    README.md
```

## Submit CLI

Run from the notebook base environment (no active conda/venv):

```bash
python jobs/submit.py --list
python jobs/submit.py sft
```

If `jobs/sft/run_uitars_lora/presubmit.py` exists, `submit.py` runs it on the NFS
checkout machine before `client_lib.Job.submit()`. Non-zero exit aborts submission.

## Registered jobs

| Submit name | Directory | `client_lib` type | Role |
|-------------|-----------|-------------------|------|
| `sft` | `sft/run_uitars_lora/` | `pytorch2` | UI-TARS LoRA SFT from pretokenized shards |

## Shared submit environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DART_JOB_IMAGE` | unset (required) | Full registry URL after `docker push` |
| `DART_USR_NAME` | `username` | Username in job description |
| `DART_PRJ_TAGS` | `#dart-gui` | Project tags in job description |
| `DART_JOB_REGION` | `SR004` | ML Space region |
| `DART_JOB_QUEUE_NAME` | unset | `queue_name`; set empty to omit |
| `DART_JOB_PRIORITY_CLASS` | `high` | Queue priority (`low`, `medium`, `high`; `none` to omit) |
| `DART_JOB` | unset | Job name if omitted on CLI |
| `DART_HF_HOME` | `data/.cache/huggingface` | NFS Hugging Face cache |
| `HF_TOKEN` | unset | Hugging Face read token (recommended) |

## Job: `sft`

Distributed UI-TARS LoRA SFT via `type="pytorch2"` and pretokenized trace shards.

See [sft/run_uitars_lora/README.md](sft/run_uitars_lora/README.md) for build, presubmit,
and worker env vars.

**Example workflow:**

```bash
# 1. Pretokenize locally
source .venv/bin/activate
python training/pretokenize_uitars_sft.py --config training/configs/sft_holo_goal_variants_example.yml

# 2. Build image
export IMAGE_NAME="job-dart-uitars-sft:1.0"
bash jobs/sft/run_uitars_lora/build.sh

# 3. Tag and push
export REGISTRY_PATH="cr.ai.cloud.ru/<workspace-uuid>"
docker tag "$IMAGE_NAME" "$REGISTRY_PATH/$IMAGE_NAME"
docker push "$REGISTRY_PATH/$IMAGE_NAME"

# 4. Submit
export DART_JOB_IMAGE="$REGISTRY_PATH/$IMAGE_NAME"
export DART_SFT_CONFIG="training/configs/sft_holo_goal_variants_example.yml"
export DART_JOB_GPUS_PER_NODE=2
python jobs/submit.py sft
```
