from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GUI_DOCKER_ENV = REPO_ROOT / "GUI-Docker-Env"
if str(GUI_DOCKER_ENV) not in sys.path:
    sys.path.insert(0, str(GUI_DOCKER_ENV))

from mm_agents.holo.schema import (  # noqa: E402
    AnswerArgs,
    ClickArgs,
    DoubleClickArgs,
    DragArgs,
    FailArgs,
    HotkeyArgs,
    PressArgs,
    ScrollArgs,
    Step,
    WaitArgs,
    WriteArgs,
)
from uitars15_format import add_box_token


def _escape_type_content(content: str) -> str:
    return content.replace("\\", "\\\\").replace("'", "\\'")


def _box(x: int, y: int) -> str:
    return f"({x},{y})"


def _normalize_hotkey(raw: str) -> str:
    keys = [part for part in re.split(r"[+,]+|\s+", raw.strip()) if part]
    return " ".join(key.lower() for key in keys)


def holo_tool_to_uitars_action(step: Step) -> str:
    tool = step.tool_call
    if isinstance(tool, ClickArgs):
        box = _box(tool.x, tool.y)
        if tool.button == "right":
            return f"right_single(start_box='{box}')"
        return f"click(start_box='{box}')"
    if isinstance(tool, DoubleClickArgs):
        return f"left_double(start_box='{_box(tool.x, tool.y)}')"
    if isinstance(tool, WriteArgs):
        content = _escape_type_content(tool.content)
        if tool.press_enter and not content.endswith("\\n"):
            content += "\\n"
        return f"type(content='{content}')"
    if isinstance(tool, HotkeyArgs):
        return f"hotkey(key='{_normalize_hotkey(tool.keys)}')"
    if isinstance(tool, PressArgs):
        return f"hotkey(key='{_normalize_hotkey(tool.key)}')"
    if isinstance(tool, ScrollArgs):
        direction = tool.direction.lower()
        if tool.x is not None and tool.y is not None:
            return f"scroll(start_box='{_box(tool.x, tool.y)}', direction='{direction}')"
        return f"scroll(start_box='(0,0)', direction='{direction}')"
    if isinstance(tool, DragArgs):
        return (
            f"drag(start_box='{_box(tool.start_x, tool.start_y)}', "
            f"end_box='{_box(tool.end_x, tool.end_y)}')"
        )
    if isinstance(tool, WaitArgs):
        return "wait()"
    if isinstance(tool, AnswerArgs):
        content = _escape_type_content(tool.content)
        return f"finished(content='{content}')"
    if isinstance(tool, FailArgs):
        return "call_user()"
    raise ValueError(f"Unsupported Holo tool: {tool}")


def parse_holo_response(raw: str) -> Step:
    return Step.model_validate_json(raw.strip())


def holo_step_to_uitars_response(step: Step) -> str:
    thought = (step.thought or "").strip()
    action = holo_tool_to_uitars_action(step)
    return add_box_token(f"Thought: {thought}\nAction: {action}")


def convert_holo_response(raw: str) -> str:
    return holo_step_to_uitars_response(parse_holo_response(raw))
