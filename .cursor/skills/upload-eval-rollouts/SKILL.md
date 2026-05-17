---
name: upload-eval-rollouts
description: Upload DART GUI evaluation rollout result directories to the tony-pitchblack/dart-gui Hugging Face Hub repo. Use when the user asks to upload inference results, publish evaluation results, or commit GUI-Docker-Env results to Hugging Face.
---

# Upload Eval Rollouts

Use this skill when asked to upload evaluation results from this repository to Hugging Face.

## Target

- Hugging Face repo: `tony-pitchblack/dart-gui-eval-rollouts`
- Reusable uploader: `GUI-Docker-Env/scripts/upload_eval_rollouts.py`
- Default discovery pattern for listing candidate folders: `GUI-Docker-Env/results*`

## Workflow

1. Confirm authentication.
   - If the HF MCP server is available, read its tool descriptor first and verify `hf_whoami`.
   - If MCP is unavailable, use the CLI instead:

```bash
source GUI-Docker-Env/.venv/bin/activate
hf auth whoami
```

2. List candidate result folders when needed:

```bash
source GUI-Docker-Env/.venv/bin/activate
python GUI-Docker-Env/scripts/upload_eval_rollouts.py --list
```

3. Upload only the folders the user asked for.
   - Pass explicit folder paths relative to the workspace root.
   - The script preserves the same folder path inside the dataset repo.

```bash
source GUI-Docker-Env/.venv/bin/activate
python GUI-Docker-Env/scripts/upload_eval_rollouts.py \
  GUI-Docker-Env/results_coact_source_readpage_verify_20260511_000535 \
  GUI-Docker-Env/results_thunderbird_all_20260510_025056
```

4. Use globs when the user specifies a family of result folders:

```bash
source GUI-Docker-Env/.venv/bin/activate
python GUI-Docker-Env/scripts/upload_eval_rollouts.py \
  --glob "GUI-Docker-Env/results_coact_uitars_gui_agent_thunderbird_*"
```

5. Dry-run before uploading if the selection is unclear:

```bash
source GUI-Docker-Env/.venv/bin/activate
python GUI-Docker-Env/scripts/upload_eval_rollouts.py \
  --dry-run \
  GUI-Docker-Env/results_coact_source_readpage_verify_20260511_000535
```

## Notes

- The script uploads each selected folder independently, so you can reuse it without syncing unrelated results.
- Missing folders fail fast by default; use `--skip-missing` only when that behavior is explicitly useful.
- `--include` and `--exclude` forward directly to the Hugging Face upload API when you want a filtered upload.

## Reporting

Report:
- which folders were uploaded
- the commit URL returned for each folder
- any missing or failed folder selections
