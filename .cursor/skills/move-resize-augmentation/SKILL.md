---
name: move-resize-augmentation
description: Create rollout-aligned move/resize screenshot perturbations from a golden GUI trace using the live emulator stack. Use when generating move_resize_augmentation data, precomputing fixed rollout geometries, writing per-step grounding prompts, or grounding captured screenshots from one golden trace.
---

# Move/Resize Augmentation

Use this skill to turn one golden GUI trace into `N` rollout-aligned screenshot tracks:

- `rollout_0001` is the default fullscreen rollout
- `rollout_0002+` are fixed move/resize perturbations sampled once per trace
- each rollout keeps the same geometry for every captured step
- the golden trace itself is still executed only once per step index

This skill builds on the live emulator workflow in [../live-emulator/SKILL.md](../live-emulator/SKILL.md). Use that skill for emulator startup, reseeding, and basic window tooling.

## Rules

- Always activate `.venv` before running the helper scripts.
- Keep rollout and step numbering 1-based with four-digit zero padding.
- Extract `per_step_prompts/` at the very start of the workflow from the golden trace plus the original golden screenshots.
- Inspect the golden `tool_events.json` and the corresponding original screenshot together before writing each prompt file.
- Prompt text must be concise and descriptive of the actual clicked UI element.
- If a step has multiple clicked UI elements, write one line per clicked element in the same `step_prompt_xxxx.txt` file.
- Generate rollout geometries once per trace. Do not resample geometry per step.
- At each step, restore the default fullscreen rollout, execute the golden action once, then capture screenshots for all rollout geometries from that resulting post-action state.
- After prompt files are written, explicitly ask the user which grounding route to use.

## Helper Scripts

- `.cursor/skills/move-resize-augmentation/scripts/trace_to_plan.py`
- `.cursor/skills/move-resize-augmentation/scripts/generate_rollout_geometries.py`
- `.cursor/skills/move-resize-augmentation/scripts/capture_step_variants.py`
- `.cursor/skills/move-resize-augmentation/scripts/write_step_prompt.py`
- `.cursor/skills/move-resize-augmentation/scripts/ground_with_api.py`

## Workflow

1. Activate the environment:

```bash
source .venv/bin/activate
```

2. Normalize the golden trace into ordered step records:

```bash
python .cursor/skills/move-resize-augmentation/scripts/trace_to_plan.py \
  --trace GUI-Docker-Env/results_.../tool_events.json \
  --output /tmp/augmentation_step_plan.json
```

3. Before starting rollout capture, inspect each click-bearing step from the normalized plan:
   - read the click action(s) from the golden trace
   - open the original `golden_screenshot_path`
   - write `per_step_prompts/step_prompt_xxxx.txt` immediately
   - use one concise line per clicked UI entity

Example prompt file content for a step with two clicked entities:

```text
the Config Editor result row in Thunderbird Settings search
the checkmark toggle for the selected preference row
```

Use the helper to write the final prompt file once the prompt text is ready:

```bash
python .cursor/skills/move-resize-augmentation/scripts/write_step_prompt.py \
  --output-root synthetic_data/thunderbird/<task_id>/move_resize_augmentation/<results_name> \
  --step-index 0006 \
  --prompt-file /tmp/step_0006_prompt.txt
```

4. Start and seed a fresh emulator using the live-emulator skill.

5. Generate rollout geometries once for the whole trace:

```bash
python .cursor/skills/move-resize-augmentation/scripts/generate_rollout_geometries.py \
  --server-port <server_port> \
  --count 2 \
  --seed 7 \
  --task thunderbird/<task_id> \
  --window-name thunderbird \
  --by-class \
  --output /tmp/rollout_geometries.json
```

6. For each step in the normalized plan:
   - restore `rollout_0001`
   - execute the golden step exactly once in that default fullscreen state
   - capture screenshots for all rollout geometries from the resulting post-action UI state
   - restore `rollout_0001` again before continuing

```bash
python .cursor/skills/move-resize-augmentation/scripts/capture_step_variants.py \
  --server-port <server_port> \
  --rollout-geometries /tmp/rollout_geometries.json \
  --output-root synthetic_data/thunderbird/<task_id>/move_resize_augmentation/<results_name> \
  --step-index 0001 \
  --actions-file /tmp/step_0001_actions.json \
  --window-name thunderbird \
  --by-class
```

7. After screenshots are captured, ask the user which grounding route to use:
   - `agent`: use the agent itself, preferably with subagents to batch rollouts or step ranges
   - `script`: run `ground_with_api.py` with env-driven OpenAI-compatible settings

```bash
python .cursor/skills/move-resize-augmentation/scripts/ground_with_api.py \
  --results-root synthetic_data/thunderbird/<task_id>/move_resize_augmentation/<results_name> \
  --prompt-dir synthetic_data/thunderbird/<task_id>/move_resize_augmentation/<results_name>/per_step_prompts
```

## Artifact Layout

```text
synthetic_data/{app_name}/{task_id}/move_resize_augmentation/{full_results_folder_name}/
  rollout_0001/
    step_0001/
      screenshot.png
      grounding.json
    step_0002/
      screenshot.png
      grounding.json
  rollout_0002/
    step_0001/
      screenshot.png
      grounding.json
  per_step_prompts/
    step_prompt_0002.txt
```

## Grounding Output Rules

- Script route writes `grounding.json` with `type: "script"` and API metadata.
- Agent route writes `grounding.json` with `type: "Cursor agent"`, `"Codex agent"`, or `"Claude agent"` and model metadata.
- If a step has no coordinate-bearing target, do not create a prompt file for it and skip grounding for that step.
- Each prompt file is newline-delimited, with one grounded UI entity description per line.

## Notes

- Prefer `rollout_0001` for the restore geometry.
- `capture_step_variants.py` now writes post-action screenshots so `step_000N` matches the original golden trajectory enumeration semantics.
- Keep the screenshot tree stable before grounding begins.
- `trace_to_plan.py` records `golden_screenshot_path` when the original trace screenshot exists, so the agent can inspect trace metadata and the exact golden screenshot together before writing prompts.
- If the user chooses the agent route, prefer parallel subagents after screenshots and prompt files already exist.
- For exact field names, geometry schema, and the grounding JSON contract, see [reference.md](reference.md).
