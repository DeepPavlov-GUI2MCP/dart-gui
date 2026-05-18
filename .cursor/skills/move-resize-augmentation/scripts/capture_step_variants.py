#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
LIVE_EMULATOR_DIR = REPO_ROOT / "GUI-Docker-Env" / "scripts" / "live_emulator"
if str(LIVE_EMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(LIVE_EMULATOR_DIR))

from common import (  # type: ignore
    DEFAULT_DESKTOP_SERVER_URL,
    DEFAULT_TOKEN,
    LiveEmulatorError,
    apply_window_geometry,
    execute_step_action,
    fetch_screenshot,
    get_emulator,
    query_window_geometry,
    write_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute one golden step in the default rollout, then capture the resulting post-action state across all fixed rollout geometries.")
    parser.add_argument("--server-url", default=DEFAULT_DESKTOP_SERVER_URL)
    parser.add_argument("--token", default=DEFAULT_TOKEN)
    parser.add_argument("--format", choices=["json", "text"], default="json")
    parser.add_argument("--emulator-id")
    parser.add_argument("--server-port", type=int)
    parser.add_argument("--rollout-geometries", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--step-index", type=int, required=True)
    parser.add_argument("--window-name", help="Window title or class to target. Defaults to active window.")
    parser.add_argument("--by-class", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--actions-file", help="Optional JSON file containing the ordered action list for this golden step.")
    parser.add_argument("--actions-json", help="Optional JSON string containing the ordered action list for this golden step.")
    parser.add_argument("--sleep-after-action", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        server_port = resolve_server_port(args)
        rollout_data = load_rollout_geometries(Path(args.rollout_geometries).expanduser().resolve())
        default_geometry = rollout_data["geometries"][0]
        target_window = resolve_target_window(
            server_port=server_port,
            window_name=args.window_name,
            by_class=args.by_class,
            timeout=args.timeout,
        )
        actions = load_actions(args)
        restore_default_geometry(
            server_port=server_port,
            target_window=target_window,
            geometry=default_geometry,
            timeout=args.timeout,
        )
        executed_actions = execute_actions(
            server_port=server_port,
            actions=actions,
            timeout=args.timeout,
            sleep_after_action=args.sleep_after_action,
        )
        captures = []
        for geometry in rollout_data["geometries"]:
            screenshot_path = write_step_capture(
                server_port=server_port,
                target_window=target_window,
                geometry=geometry,
                output_root=Path(args.output_root).expanduser().resolve(),
                step_index=args.step_index,
                timeout=args.timeout,
                overwrite=args.overwrite,
            )
            captures.append(
                {
                    "rollout_index": geometry["rollout_index"],
                    "rollout_name": geometry["rollout_name"],
                    "screenshot_path": str(screenshot_path),
                }
            )
        restore_default_geometry(
            server_port=server_port,
            target_window=target_window,
            geometry=default_geometry,
            timeout=args.timeout,
        )
        payload = {
            "status": "ok",
            "step_index": args.step_index,
            "executed_action_count": executed_actions,
            "capture_count": len(captures),
            "target_window": {
                "window_id": target_window.window_id,
                "title": target_window.title,
                "wm_class": target_window.wm_class,
            },
            "captures": captures,
        }
        write_output(payload, args.format)
        return 0
    except LiveEmulatorError as exc:
        if args.format == "json":
            write_output({"status": "error", "error": str(exc)}, "json")
        else:
            print(f"error: {exc}")
        return 1


def resolve_server_port(args: argparse.Namespace) -> int:
    if bool(args.emulator_id) == bool(args.server_port):
        raise LiveEmulatorError("Provide exactly one of --emulator-id or --server-port.")
    if args.by_class and not args.window_name:
        raise LiveEmulatorError("--by-class requires --window-name.")
    if args.server_port:
        return args.server_port
    emulator = get_emulator(args.server_url, args.emulator_id, args.token)
    return emulator.server_port


def load_actions(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.actions_file and args.actions_json:
        raise LiveEmulatorError("Provide at most one of --actions-file or --actions-json.")
    if args.actions_file:
        raw = json.loads(Path(args.actions_file).expanduser().resolve().read_text(encoding="utf-8"))
    elif args.actions_json:
        raw = json.loads(args.actions_json)
    else:
        return []
    if not isinstance(raw, list):
        raise LiveEmulatorError("Expected an action list in --actions-file/--actions-json.")
    actions: list[dict[str, Any]] = []
    for action in raw:
        if not isinstance(action, dict):
            raise LiveEmulatorError("Each action must be a JSON object.")
        actions.append(action)
    return actions


def load_rollout_geometries(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LiveEmulatorError(f"Expected JSON object in {path}")
    geometries = payload.get("geometries")
    if not isinstance(geometries, list) or not geometries:
        raise LiveEmulatorError(f"Missing geometries in {path}")
    return payload


def write_step_capture(
    *,
    server_port: int,
    target_window: Any,
    geometry: dict[str, Any],
    output_root: Path,
    step_index: int,
    timeout: int,
    overwrite: bool,
) -> Path:
    apply_geometry(
        server_port=server_port,
        target_window=target_window,
        geometry=geometry,
        timeout=timeout,
    )
    step_dir = output_root / geometry["rollout_name"] / f"step_{step_index:04d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = step_dir / "screenshot.png"
    if screenshot_path.exists() and not overwrite:
        return screenshot_path
    screenshot_path.write_bytes(fetch_screenshot(server_port))
    return screenshot_path


def execute_actions(
    *,
    server_port: int,
    actions: list[dict[str, Any]],
    timeout: int,
    sleep_after_action: float,
) -> int:
    for action in actions:
        result = execute_step_action(
            server_port,
            action,
            timeout=max(timeout, 90),
            sleep_after=sleep_after_action,
        )
        if int(result.get("returncode", 0)) != 0:
            raise LiveEmulatorError(f"Golden action execution failed: {result}")
    return len(actions)


def restore_default_geometry(
    *,
    server_port: int,
    target_window: Any,
    geometry: dict[str, Any],
    timeout: int,
) -> None:
    apply_geometry(
        server_port=server_port,
        target_window=target_window,
        geometry=geometry,
        timeout=timeout,
    )


def apply_geometry(
    *,
    server_port: int,
    target_window: Any,
    geometry: dict[str, Any],
    timeout: int,
) -> None:
    apply_window_geometry(
        server_port,
        target_window,
        x=int(geometry["x"]),
        y=int(geometry["y"]),
        width=int(geometry["width"]),
        height=int(geometry["height"]),
        timeout=timeout,
    )


def resolve_target_window(
    *,
    server_port: int,
    window_name: str | None,
    by_class: bool,
    timeout: int,
) -> Any:
    active_window = query_window_geometry(server_port, timeout=timeout)
    if not window_name:
        return active_window
    if matches_target(active_window, window_name=window_name, by_class=by_class):
        return active_window
    return query_window_geometry(
        server_port,
        window_name=window_name,
        by_class=by_class,
        timeout=timeout,
    )


def matches_target(window: Any, *, window_name: str, by_class: bool) -> bool:
    needle = window_name.lower()
    if by_class:
        class_parts = [part.lower() for part in getattr(window, "wm_class", [])]
        return needle in class_parts or any(needle in part for part in class_parts)
    return needle in str(getattr(window, "title", "")).lower()


if __name__ == "__main__":
    raise SystemExit(main())
