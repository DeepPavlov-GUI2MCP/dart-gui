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

### Codex CLI Agent Route

Prefer this route when the user explicitly wants Codex grounding or wants one fresh
chat per image. Use one non-interactive `codex exec` call per screenshot and write
the result back to the rollout step directory.

Rules for this route:

- Use `codex -a never exec`, not `codex exec --ask-for-approval ...`; the approval flag
  belongs before the `exec` subcommand.
- Pass the prompt on stdin with a final `-` argument. Do not rely on positional prompt
  text after the image args; stdin is the stable path.
- Use `--ephemeral` so each screenshot gets a brand new chat with no carry-over context.
- Use `--sandbox read-only` because grounding should not execute write actions.
- Use `--output-last-message <file>` and parse JSON from that file rather than trying to
  recover the final answer from stderr/stdout logs.
- Ask for `x_norm` / `y_norm` relative to the attached image.
- For post-action grounding, write `grounding.json` in `rollout_%04d/step_%04d/`.
- For pre-action grounding, write `pre_action_grounding.json` in `rollout_%04d/step_%04d/`
  and ground step `N` on screenshot `step_(N-1)`.

Recommended exact args for one screenshot:

```bash
codex -a never exec \
  --ephemeral \
  --sandbox read-only \
  --model gpt-5.4 \
  --output-last-message /tmp/codex_ground_step_0008.txt \
  --image synthetic_data/thunderbird/<task_id>/move_resize_augmentation/<results_name>/rollout_0001/step_0007/screenshot.png \
  -
```

Recommended stdin prompt template:

```text
You are grounding UI elements in a pre-action screenshot.
Return JSON only, with no markdown.
The screenshot shows the UI state immediately before the action for step 0008.
The target descriptions are newline-delimited below.
If there is one target, return an object with keys x_norm, y_norm, element_description, confidence.
If there are multiple targets, return an object with a targets array containing one object per target in the same order, each with x_norm, y_norm, element_description, confidence.
Coordinates must be normalized to [0,1] relative to the full image.
Do not add any prose.
Target description(s):
the click point inside the about:config search field at the top of the Advanced Preferences tab, over the existing query text around one quarter of the way across the long input; target the text-entry area near x=528 in the fullscreen screenshot, not the left padding and not the field center
```

Recommended loop for pre-action grounding on one rollout:

```bash
source .venv/bin/activate && python - <<'PY'
from pathlib import Path
import json
import subprocess
import textwrap

root = Path("synthetic_data/thunderbird/<task_id>/move_resize_augmentation/<results_name>")
rollout_dir = root / "rollout_0001"
prompt_dir = root / "per_step_prompts"
model = "gpt-5.4"

def extract_json(text: str):
    text = text.strip()
    candidates = [text]
    if "```" in text:
        for chunk in text.split("```"):
            chunk = chunk.strip()
            if not chunk:
                continue
            if chunk.startswith("json"):
                chunk = chunk[4:].strip()
            candidates.append(chunk)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            pass
    raise ValueError(f"Could not parse JSON from: {text[:400]}")

for prompt_path in sorted(prompt_dir.glob("step_prompt_*.txt")):
    step_index = int(prompt_path.stem.split("_")[-1])
    pre_step_index = step_index - 1
    screenshot_path = rollout_dir / f"step_{pre_step_index:04d}" / "screenshot.png"
    output_step_dir = rollout_dir / f"step_{step_index:04d}"
    grounding_path = output_step_dir / "pre_action_grounding.json"
    raw_path = Path("/tmp") / f"codex_rollout0001_gpt54_preaction_step_{step_index:04d}.txt"
    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    user_prompt = textwrap.dedent(f"""\
You are grounding UI elements in a pre-action screenshot.
Return JSON only, with no markdown.
The screenshot shows the UI state immediately before the action for step {step_index:04d}.
The target descriptions are newline-delimited below.
If there is one target, return an object with keys x_norm, y_norm, element_description, confidence.
If there are multiple targets, return an object with a targets array containing one object per target in the same order, each with x_norm, y_norm, element_description, confidence.
Coordinates must be normalized to [0,1] relative to the full image.
Do not add any prose.
Target description(s):
{prompt_text}
""")
    cmd = [
        "codex", "-a", "never", "exec",
        "--ephemeral",
        "--sandbox", "read-only",
        "--model", model,
        "--output-last-message", str(raw_path),
        "--image", str(screenshot_path),
        "-",
    ]
    result = subprocess.run(cmd, input=user_prompt, text=True, capture_output=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(
            f"step {step_index} failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    parsed = extract_json(raw_path.read_text(encoding="utf-8"))
    payload = {
        "type": "Codex agent",
        "model": model,
        "grounding_mode": "pre_action",
        "step_index": step_index,
        "pre_action_step_index": pre_step_index,
        "rollout_index": 1,
        "prompt_file": str(prompt_path.relative_to(root)),
        "prompt_inline": prompt_text,
        "screenshot_path": str(screenshot_path.relative_to(root)),
        "output_step_dir": str(output_step_dir.relative_to(root)),
        "raw_response": parsed,
    }
    if isinstance(parsed, dict) and "targets" in parsed:
        payload["targets"] = parsed["targets"]
    elif isinstance(parsed, dict):
        payload.update(parsed)
    else:
        payload["parsed"] = parsed
    grounding_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"grounded step {step_index:04d}")
PY
```

Post-action grounding uses the same Codex args and parsing logic, but swaps:

- `screenshot_path = rollout_dir / f"step_{step_index:04d}" / "screenshot.png"`
- `grounding_path = output_step_dir / "grounding.json"`
- prompt wording should describe the visible post-action state, not the vanished pre-click control

When debugging this route, the most common failure modes are:

- prompt references a control that disappeared after the click
- prompt names the right widget but not the intended point inside it
- `codex exec` prompt is passed incorrectly instead of via stdin
- raw coordinates are compared across rollouts without converting to window-relative space

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
