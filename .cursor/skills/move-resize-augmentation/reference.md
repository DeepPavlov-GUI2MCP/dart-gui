# Reference

## Indexing

- Rollouts are 1-based.
- Steps are 1-based.
- Rollout directory names use `rollout_%04d`.
- Step directory names use `step_%04d`.
- Prompt files use `step_prompt_%04d.txt`.

Examples:

- `rollout_0001`
- `rollout_0002`
- `step_0001`
- `step_prompt_0008.txt`

## Normalized Step Plan

`trace_to_plan.py` emits one ordered record per `computer_call` step:

```json
{
  "step_index": 8,
  "response_id": "resp_...",
  "call_id": "call_...",
  "golden_screenshot_path": "/abs/path/to/step_8.png",
  "action_types": ["click", "keypress", "type"],
  "requires_grounding": true,
  "actions": [
    {"type": "click", "x": 528, "y": 174, "button": "left"},
    {"type": "keypress", "keys": ["CTRL", "A"]},
    {"type": "type", "text": "mail.imap.use_status_for_biff"}
  ],
  "click_actions": [
    {"action_index": 1, "x": 528, "y": 174, "button": "left"}
  ],
  "summary": "click(528,174); keypress(['CTRL', 'A']); type('mail.imap.use_status_for_biff')"
}
```

`requires_grounding` is `true` when the step contains at least one click action with coordinates.
`golden_screenshot_path` is populated when the original trace directory contains `step_{step_index}.png`.

## Rollout Geometry File

`generate_rollout_geometries.py` writes a single JSON document for the whole trace:

```json
{
  "count": 2,
  "seed": 7,
  "default_rollout_index": 1,
  "geometries": [
    {
      "rollout_index": 1,
      "rollout_name": "rollout_0001",
      "is_default": true,
      "x": 0,
      "y": 0,
      "width": 1920,
      "height": 1080
    },
    {
      "rollout_index": 2,
      "rollout_name": "rollout_0002",
      "is_default": false,
      "x": 120,
      "y": 60,
      "width": 1420,
      "height": 920
    }
  ]
}
```

The default rollout is the current target window geometry when the script runs. Non-default rollouts are sampled once and then reused across every step capture.

## Screenshot Layout

All screenshots live under the chosen results root:

```text
synthetic_data/{app_name}/{task_id}/move_resize_augmentation/{full_results_folder_name}/
  rollout_0001/
    step_0001/
      screenshot.png
    step_0002/
      screenshot.png
  rollout_0002/
    step_0001/
      screenshot.png
```

Each rollout directory should read like one coherent trace collected under one persistent geometry.
Each `step_%04d/screenshot.png` is the post-action state after the golden step of the same index has been executed once in `rollout_0001`, matching the original trajectory screenshot numbering.

## Prompt Files

Prompt files are written at the start of the procedure, before rollout capture begins.

Rules:

- Write `per_step_prompts/step_prompt_%04d.txt` only when a step needs coordinate grounding.
- Build each prompt file by inspecting the golden trace action(s) and the original golden screenshot together.
- The prompt text should describe the actual clicked UI element, not the raw coordinates.
- If a step contains multiple click actions, write one concise element description per line.
- Skip prompt creation for screenshot-only, typing-only, or keypress-only steps.
- Keep prompt text focused on the UI element to ground in the screenshot, not on replaying the whole task.

Example:

```text
the Thunderbird Settings button in the lower-left navigation area
the Config Editor search result row
```

## Grounding Routes

The user must explicitly choose one route after screenshots and prompt files are ready.

### Agent route

- Prefer batching with subagents once the artifacts are already on disk.
- Save one `grounding.json` per rollout-step directory.
- Set:
  - `type` to `"Cursor agent"`, `"Codex agent"`, or `"Claude agent"`
  - `model` to the model identity actually used

### Script route

Use `ground_with_api.py` with these env vars:

- `GROUNDING_API_KEY` required
- `GROUNDING_MODEL` required
- `GROUNDING_BASE_URL` optional
- `GROUNDING_TIMEOUT` optional, default `60`
- `GROUNDING_MAX_RETRIES` optional, default `3`

The script writes one `grounding.json` per rollout-step directory.

## grounding.json Schema

Both routes should preserve a common shape:

```json
{
  "type": "script",
  "model": "gpt-4.1-mini",
  "base_url": "http://127.0.0.1:8010/v1",
  "step_index": 8,
  "rollout_index": 2,
  "prompt_file": "per_step_prompts/step_prompt_0008.txt",
  "screenshot_path": "rollout_0002/step_0008/screenshot.png",
  "element_description": "the search field in the config editor toolbar",
  "x_norm": 0.271,
  "y_norm": 0.162,
  "confidence": 0.93
}
```

Required fields:

- `type`
- `model`
- `step_index`
- `rollout_index`
- `screenshot_path`
- `x_norm`
- `y_norm`

Optional fields:

- `base_url`
- `prompt_file`
- `element_description`
- `confidence`
- any provider-specific raw response block

## Retry Guidance

- Fail fast on malformed geometry files or missing screenshot/prompt paths.
- For screenshot capture, stop immediately if the target window cannot be found or restored.
- For the script grounding route, retry API calls up to `GROUNDING_MAX_RETRIES`.
- Do not mutate or overwrite screenshots during grounding.

## Capture Semantics

Per step:

1. Restore `rollout_0001` / default geometry.
2. Execute the golden step actions exactly once there.
3. Capture the resulting post-action state under every rollout geometry.
4. Restore `rollout_0001` again before the next step.

Example invocation:

```bash
python .cursor/skills/move-resize-augmentation/scripts/capture_step_variants.py \
  --server-port <server_port> \
  --rollout-geometries /tmp/rollout_geometries.json \
  --output-root synthetic_data/thunderbird/<task_id>/move_resize_augmentation/<results_name> \
  --step-index 0003 \
  --actions-file /tmp/step_0003_actions.json \
  --window-name thunderbird \
  --by-class
```
