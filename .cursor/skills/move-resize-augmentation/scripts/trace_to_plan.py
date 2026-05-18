#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize a tool_events trace into ordered step records.")
    parser.add_argument("--trace", required=True, help="Path to tool_events.json")
    parser.add_argument("--output", help="Optional output path for the normalized step plan.")
    parser.add_argument("--format", choices=["json", "text"], default="json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trace_path = Path(args.trace).expanduser().resolve()
    plan = normalize_trace(trace_path)
    payload: dict[str, Any] = {
        "trace_path": str(trace_path),
        "step_count": len(plan),
        "steps": plan,
    }
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(f"trace_path: {trace_path}")
        print(f"step_count: {len(plan)}")
        for step in plan:
            print(f"step_{step['step_index']:04d}: {step['summary']}")
    return 0


def normalize_trace(trace_path: Path) -> list[dict[str, Any]]:
    raw = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected list in {trace_path}")
    trace_dir = trace_path.parent
    steps: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "computer_call":
            continue
        step_index = int(item["step_index"])
        actions = [normalize_action(action) for action in item.get("actions", []) if isinstance(action, dict)]
        click_actions = []
        for action_index, action in enumerate(actions, start=1):
            if action.get("type") != "click":
                continue
            if "x" not in action or "y" not in action:
                continue
            click_actions.append(
                {
                    "action_index": action_index,
                    "x": int(action["x"]),
                    "y": int(action["y"]),
                    "button": str(action.get("button") or "left"),
                }
            )
        steps.append(
            {
                "step_index": step_index,
                "response_id": item.get("response_id"),
                "call_id": item.get("call_id"),
                "golden_screenshot_path": resolve_golden_screenshot_path(trace_dir, step_index),
                "action_types": [str(action.get("type", "")) for action in actions],
                "requires_grounding": bool(click_actions),
                "actions": actions,
                "click_actions": click_actions,
                "summary": summarize_actions(actions),
            }
        )
    steps.sort(key=lambda step: int(step["step_index"]))
    return steps


def normalize_action(action: dict[str, Any]) -> dict[str, Any]:
    action_type = str(action.get("type", "")).strip().lower()
    if action_type == "click":
        return {
            "type": "click",
            "x": int(action["x"]),
            "y": int(action["y"]),
            "button": str(action.get("button") or "left"),
        }
    if action_type == "type":
        return {"type": "type", "text": str(action.get("text", ""))}
    if action_type == "keypress":
        return {"type": "keypress", "keys": [str(key) for key in (action.get("keys") or [])]}
    if action_type == "screenshot":
        return {"type": "screenshot"}
    return {"type": action_type or "unknown", **action}


def summarize_actions(actions: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for action in actions:
        action_type = action.get("type")
        if action_type == "click":
            parts.append(f"click({action['x']},{action['y']})")
            continue
        if action_type == "type":
            parts.append(f"type({action['text']!r})")
            continue
        if action_type == "keypress":
            parts.append(f"keypress({action['keys']!r})")
            continue
        parts.append(str(action_type))
    return "; ".join(parts) if parts else "(no actions)"


def resolve_golden_screenshot_path(trace_dir: Path, step_index: int) -> str | None:
    candidate = trace_dir / f"step_{step_index}.png"
    if candidate.is_file():
        return str(candidate.resolve())
    return None


if __name__ == "__main__":
    raise SystemExit(main())
