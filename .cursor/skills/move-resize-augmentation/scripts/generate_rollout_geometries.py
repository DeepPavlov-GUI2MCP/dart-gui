#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
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
    clamp_window_size,
    compute_coordinate_limits,
    fetch_screen_size,
    get_emulator,
    query_window_geometry,
    resolve_geometry_threshold,
    write_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed rollout geometries for one golden trace.")
    parser.add_argument("--server-url", default=DEFAULT_DESKTOP_SERVER_URL)
    parser.add_argument("--token", default=DEFAULT_TOKEN)
    parser.add_argument("--format", choices=["json", "text"], default="json")
    parser.add_argument("--emulator-id")
    parser.add_argument("--server-port", type=int)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", help="Task ref like thunderbird/<task_id> for threshold selection.")
    parser.add_argument("--app", help="App name override for threshold selection.")
    parser.add_argument("--window-name", help="Window title or class to target. Defaults to active window.")
    parser.add_argument("--by-class", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.count <= 0:
            raise LiveEmulatorError("--count must be greater than zero.")
        server_port = resolve_server_port(args)
        default_window = resolve_target_window(
            server_port=server_port,
            window_name=args.window_name,
            by_class=args.by_class,
            timeout=args.timeout,
        )
        screen = fetch_screen_size(server_port)
        threshold = resolve_geometry_threshold(task=args.task, app=args.app)
        geometries = build_rollout_geometries(
            count=args.count,
            seed=args.seed,
            default_window=default_window,
            screen=screen.to_dict(),
            threshold=threshold,
        )
        restore_default_geometry(server_port, default_window, timeout=args.timeout)
        payload = {
            "status": "ok",
            "count": args.count,
            "seed": args.seed,
            "default_rollout_index": 1,
            "screen": screen.to_dict(),
            "threshold": threshold.to_dict(),
            "geometries": geometries,
        }
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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


def build_rollout_geometries(
    *,
    count: int,
    seed: int,
    default_window: Any,
    screen: dict[str, int],
    threshold: Any,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    geometries = [
        {
            "rollout_index": 1,
            "rollout_name": "rollout_0001",
            "is_default": True,
            "x": int(default_window.x),
            "y": int(default_window.y),
            "width": int(default_window.width),
            "height": int(default_window.height),
        }
    ]
    screen_size = type("ScreenProxy", (), screen)()
    seen = {(default_window.x, default_window.y, default_window.width, default_window.height)}
    for rollout_index in range(2, count + 1):
        geometry = sample_geometry(rng, screen_size, threshold, seen)
        geometries.append(
            {
                "rollout_index": rollout_index,
                "rollout_name": f"rollout_{rollout_index:04d}",
                "is_default": False,
                **geometry,
            }
        )
    return geometries


def sample_geometry(
    rng: random.Random,
    screen: Any,
    threshold: Any,
    seen: set[tuple[int, int, int, int]],
) -> dict[str, int]:
    initial_bounds = compute_coordinate_limits(
        screen,
        threshold,
        width=threshold.min_width,
        height=threshold.min_height,
    )
    for _ in range(100):
        width = rng.randint(initial_bounds.min_width, initial_bounds.max_width)
        height = rng.randint(initial_bounds.min_height, initial_bounds.max_height)
        width, height, _ = clamp_window_size(width, height, screen, threshold, allow_below_min=False)
        bounds = compute_coordinate_limits(screen, threshold, width=width, height=height)
        x = rng.randint(bounds.min_x, bounds.max_x)
        y = rng.randint(bounds.min_y, bounds.max_y)
        key = (x, y, width, height)
        if key in seen:
            continue
        seen.add(key)
        return {"x": x, "y": y, "width": width, "height": height}
    raise LiveEmulatorError("Could not generate a unique rollout geometry.")


def restore_default_geometry(server_port: int, window: Any, *, timeout: int) -> None:
    apply_window_geometry(
        server_port,
        window,
        x=int(window.x),
        y=int(window.y),
        width=int(window.width),
        height=int(window.height),
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
