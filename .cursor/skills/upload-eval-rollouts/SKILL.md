---
name: upload-eval-rollouts
description: Upload DART GUI evaluation rollout result directories to the tony-pitchblack/dart-gui Hugging Face Hub repo. Use when the user asks to upload inference results, publish evaluation results, or commit GUI-Docker-Env results to Hugging Face.
---

# Upload Eval Rollouts

Use this skill when asked to upload inference results from this repository to Hugging Face.

## Target

- Hugging Face repo: `tony-pitchblack/dart-gui` (dataset repo; pass `--repo-type dataset` to `hf upload`, since the CLI default is `model`)
- Result directories:
  - `GUI-Docker-Env/results`
  - `GUI-Docker-Env/results_single_proxy`
  - `GUI-Docker-Env/results_single_task_watch`

## Workflow

1. Confirm Hugging Face authentication with the HF MCP server first.
   - Read the MCP tool descriptor before calling any MCP tool.
   - Call `hf_whoami` and verify the authenticated user is `tony-pitchblack`.
   - The current HF MCP server is read-only; use it for auth/query verification, then use `hf upload` for the write commit unless a write-capable MCP upload tool is added.
2. Verify the three local result directories exist.
3. Activate the local environment before terminal commands:

```bash
source GUI-Docker-Env/.venv/bin/activate
```

4. Upload each directory with `hf upload`, preserving the same paths in the repo:

```bash
hf upload tony-pitchblack/dart-gui GUI-Docker-Env/results GUI-Docker-Env/results --repo-type dataset --commit-message "Upload inference results"
hf upload tony-pitchblack/dart-gui GUI-Docker-Env/results_single_proxy GUI-Docker-Env/results_single_proxy --repo-type dataset --commit-message "Upload single proxy inference results"
hf upload tony-pitchblack/dart-gui GUI-Docker-Env/results_single_task_watch GUI-Docker-Env/results_single_task_watch --repo-type dataset --commit-message "Upload single task watch inference results"
```

5. If a result directory is large or the upload is interrupted, retry that directory with `hf upload-large-folder`.

## Reporting

Report which directories uploaded successfully and mention any directory that was missing or failed.
