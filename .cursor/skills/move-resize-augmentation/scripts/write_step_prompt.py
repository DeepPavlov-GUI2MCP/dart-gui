#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write one per-step grounding prompt file.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--step-index", type=int, required=True)
    parser.add_argument("--prompt", help="Prompt text to write. If empty, the file is skipped.")
    parser.add_argument("--prompt-file", help="Optional file to read prompt text from.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--format", choices=["json", "text"], default="json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompt_lines = normalize_prompt_lines(resolve_prompt_text(args))
    if not prompt_lines:
        payload = {
            "status": "skipped",
            "step_index": args.step_index,
            "reason": "empty_prompt",
        }
        emit(payload, args.format)
        return 0
    prompt_dir = Path(args.output_root).expanduser().resolve() / "per_step_prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompt_dir / f"step_prompt_{args.step_index:04d}.txt"
    if prompt_path.exists() and not args.overwrite:
        payload = {
            "status": "skipped",
            "step_index": args.step_index,
            "prompt_path": str(prompt_path),
            "reason": "exists",
        }
        emit(payload, args.format)
        return 0
    prompt_path.write_text("\n".join(prompt_lines) + "\n", encoding="utf-8")
    payload = {
        "status": "ok",
        "step_index": args.step_index,
        "prompt_path": str(prompt_path),
        "line_count": len(prompt_lines),
    }
    emit(payload, args.format)
    return 0


def resolve_prompt_text(args: argparse.Namespace) -> str:
    if bool(args.prompt) == bool(args.prompt_file):
        raise ValueError("Provide exactly one of --prompt or --prompt-file.")
    if args.prompt_file:
        return Path(args.prompt_file).expanduser().resolve().read_text(encoding="utf-8")
    return str(args.prompt or "")


def normalize_prompt_lines(prompt_text: str) -> list[str]:
    lines = [line.strip() for line in prompt_text.splitlines()]
    return [line for line in lines if line]


def emit(payload: dict[str, object], output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(payload, indent=2))
        return
    for key, value in payload.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    raise SystemExit(main())
