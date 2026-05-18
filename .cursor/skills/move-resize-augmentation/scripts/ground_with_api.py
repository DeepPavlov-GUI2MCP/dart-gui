#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ground captured rollout screenshots with an OpenAI-compatible API.")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--prompt-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--format", choices=["json", "text"], default="json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config_from_env()
    results_root = Path(args.results_root).expanduser().resolve()
    prompt_dir = Path(args.prompt_dir).expanduser().resolve()
    records = []
    for prompt_path in sorted(prompt_dir.glob("step_prompt_*.txt")):
        step_index = parse_step_index(prompt_path)
        prompt_text = prompt_path.read_text(encoding="utf-8").strip()
        if not prompt_text:
            continue
        for rollout_dir in sorted(results_root.glob("rollout_*")):
            if not rollout_dir.is_dir():
                continue
            rollout_index = parse_rollout_index(rollout_dir)
            step_dir = rollout_dir / f"step_{step_index:04d}"
            screenshot_path = step_dir / "screenshot.png"
            grounding_path = step_dir / "grounding.json"
            if not screenshot_path.is_file():
                continue
            if grounding_path.exists() and not args.overwrite:
                records.append(
                    {
                        "status": "skipped",
                        "rollout_index": rollout_index,
                        "step_index": step_index,
                        "grounding_path": str(grounding_path),
                        "reason": "exists",
                    }
                )
                continue
            grounding = request_grounding(
                screenshot_bytes=screenshot_path.read_bytes(),
                prompt=prompt_text,
                config=config,
            )
            payload = {
                "type": "script",
                "model": config["model"],
                "base_url": config.get("base_url"),
                "step_index": step_index,
                "rollout_index": rollout_index,
                "prompt_file": str(prompt_path.relative_to(results_root)),
                "screenshot_path": str(screenshot_path.relative_to(results_root)),
                **grounding,
            }
            step_dir.mkdir(parents=True, exist_ok=True)
            grounding_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            records.append(
                {
                    "status": "ok",
                    "rollout_index": rollout_index,
                    "step_index": step_index,
                    "grounding_path": str(grounding_path),
                }
            )
    emit(
        {
            "status": "ok",
            "results_root": str(results_root),
            "prompt_dir": str(prompt_dir),
            "processed": len(records),
            "records": records,
        },
        args.format,
    )
    return 0


def load_config_from_env() -> dict[str, Any]:
    api_key = (os.getenv("GROUNDING_API_KEY") or "").strip()
    model = (os.getenv("GROUNDING_MODEL") or "").strip()
    base_url = (os.getenv("GROUNDING_BASE_URL") or "").strip()
    timeout = float((os.getenv("GROUNDING_TIMEOUT") or "60").strip())
    max_retries = int((os.getenv("GROUNDING_MAX_RETRIES") or "3").strip())
    organization = (os.getenv("GROUNDING_ORGANIZATION") or "").strip() or None
    project = (os.getenv("GROUNDING_PROJECT") or "").strip() or None
    missing = [name for name, value in (("GROUNDING_API_KEY", api_key), ("GROUNDING_MODEL", model)) if not value]
    if missing:
        raise ValueError(f"Missing required grounding env vars: {', '.join(missing)}")
    return {
        "api_key": api_key,
        "model": model,
        "base_url": base_url or "https://api.openai.com/v1",
        "timeout": timeout,
        "max_retries": max(1, max_retries),
        "organization": organization,
        "project": project,
    }


def parse_step_index(prompt_path: Path) -> int:
    stem = prompt_path.stem
    return int(stem.split("_")[-1])


def parse_rollout_index(rollout_dir: Path) -> int:
    return int(rollout_dir.name.split("_")[-1])


def request_grounding(*, screenshot_bytes: bytes, prompt: str, config: dict[str, Any]) -> dict[str, Any]:
    body = {
        "model": config["model"],
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_bytes_to_data_url(screenshot_bytes),
                        },
                    },
                ],
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }
    if config.get("organization"):
        headers["OpenAI-Organization"] = str(config["organization"])
    if config.get("project"):
        headers["OpenAI-Project"] = str(config["project"])
    endpoint = config["base_url"].rstrip("/") + "/chat/completions"
    last_error: Exception | None = None
    for _ in range(config["max_retries"]):
        try:
            response = requests.post(endpoint, headers=headers, json=body, timeout=config["timeout"])
            response.raise_for_status()
            payload = response.json()
            return parse_grounding_response(extract_completion_text(payload))
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Grounding API request failed: {last_error}") from last_error


def image_bytes_to_data_url(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def extract_completion_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"Unexpected completion payload: {payload!r}")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(part.strip() for part in parts if part.strip()).strip()
    return str(content).strip()


def parse_grounding_response(response_text: str) -> dict[str, Any]:
    cleaned = response_text.strip()
    if not cleaned:
        raise ValueError("Grounding API returned empty content.")
    candidates = [cleaned]
    if "```" in cleaned:
        for chunk in cleaned.split("```"):
            chunk = chunk.strip()
            if not chunk:
                continue
            if chunk.startswith("json"):
                chunk = chunk[4:].strip()
            candidates.append(chunk)
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        x_norm = payload.get("x_norm")
        y_norm = payload.get("y_norm")
        if x_norm is None or y_norm is None:
            raise ValueError(f"Grounding response missing x_norm/y_norm: {payload!r}")
        result = dict(payload)
        result["x_norm"] = float(x_norm)
        result["y_norm"] = float(y_norm)
        if not (0.0 <= result["x_norm"] <= 1.0 and 0.0 <= result["y_norm"] <= 1.0):
            raise ValueError(f"Grounding coordinates must be normalized to [0,1]: {payload!r}")
        return result
    raise ValueError(f"Could not parse grounding response as JSON: {response_text!r}")


def emit(payload: dict[str, Any], output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(payload, indent=2))
        return
    for key, value in payload.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    raise SystemExit(main())
