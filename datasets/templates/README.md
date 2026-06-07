# SFT Dataset Templates

## UI-TARS trace SFT

Writer/eval rollouts under `GUI-Docker-Env/results_uitars15_writer_*` contain:

- `traj.jsonl` with `step_num`, `response`, and `screenshot_file`
- PNG screenshots beside the trajectory (`screenshot_file` for step 1 is the initial desktop state)
- `result.txt` with the evaluator score

Point `train_sft.dataset.format: uitars_trace` at those rollout roots and set `task_examples_dir` to `GUI-Docker-Env/evaluation_examples/examples`. Training code stages cached JSONL under `datasets/runtime/uitars-traces-*`.

See `training/configs/sft_example.yml` and `.cursor/skills/train-sft/SKILL.md`.

## Holo trace SFT

Holo eval rollouts under `GUI-Docker-Env/results_holo_gpt54_writer_traces_*` include preflight artifact rows plus agent steps with Holo Step JSON in `response`.

Set `trace_source: holo` (or `auto`) and use `training/configs/sft_holo_example.yml`. The builder converts Holo JSON to UI-TARS labels and maps pre-action screenshots across preflight and post-action agent rows.

Trace-collection runs often have `result.txt: -1`; leave `min_result` unset until eval scores are populated.

## Legacy chat JSONL

`sft_messages.jsonl` uses OpenAI-style `messages` records for TRL chat SFT.

Upload this format to a Hugging Face dataset repo, set `train_sft.dataset.format: chat`, and configure `train_sft.dataset.repo_id` (see `training/configs/sft_example_chat.yml`).
