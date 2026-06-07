# SFT Dataset Templates

## UI-TARS trace SFT

Writer/eval rollouts under `GUI-Docker-Env/results_uitars15_writer_*` contain:

- `traj.jsonl` with `step_num`, `response`, and `screenshot_file`
- PNG screenshots beside the trajectory
- `result.txt` with the evaluator score

Point `train_sft.dataset.format: uitars_trace` at those rollout roots and set `task_examples_dir` to `GUI-Docker-Env/evaluation_examples/examples`. Training code stages cached JSONL under `datasets/runtime/uitars-traces-*`.

See `training/configs/sft_example.yml` and `.cursor/skills/train-sft/SKILL.md`.

## Legacy chat JSONL

`sft_messages.jsonl` uses OpenAI-style `messages` records for TRL chat SFT.

Upload this format to a Hugging Face dataset repo, set `train_sft.dataset.format: chat`, and configure `train_sft.dataset.repo_id` (see `training/configs/sft_example_chat.yml`).
