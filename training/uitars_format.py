from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
GUI_DOCKER_ENV = REPO_ROOT / "GUI-Docker-Env"
if str(GUI_DOCKER_ENV) not in sys.path:
    sys.path.insert(0, str(GUI_DOCKER_ENV))

from mm_agents.uitars15_v1 import (  # noqa: E402
    UITARS_NORMAL_ACTION_SPACE,
    UITARS_USR_PROMPT_THOUGHT,
    add_box_token,
    pil_to_base64,
)


@dataclass(frozen=True)
class UitarsFormatSettings:
    prompt_style: str = "qwen25vl_normal"
    infer_mode: str = "qwen25vl_normal"
    language: str = "English"
    max_pixels: int = 16384 * 28 * 28
    min_pixels: int = 100 * 28 * 28
    history_n: int = 5


def resolve_action_space(settings: UitarsFormatSettings) -> str:
    if settings.infer_mode == "qwen25vl_normal":
        return UITARS_NORMAL_ACTION_SPACE
    raise ValueError(f"Unsupported infer_mode: {settings.infer_mode}")


def resolve_prompt_template(settings: UitarsFormatSettings) -> str:
    if settings.prompt_style in {"qwen2vl_user", "qwen25vl_normal"}:
        return UITARS_USR_PROMPT_THOUGHT
    raise ValueError(f"Unsupported prompt_style: {settings.prompt_style}")


def load_pil_image(source: bytes | Path) -> Image.Image:
    if isinstance(source, Path):
        with open(source, "rb") as f:
            source = f.read()
    image = Image.open(BytesIO(source))
    return image


def resize_pil_image(image: Image.Image, *, max_pixels: int, min_pixels: int) -> Image.Image:
    if image.width * image.height > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width = int(image.width * resize_factor)
        height = int(image.height * resize_factor)
        image = image.resize((width, height))
    if image.width * image.height < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width = math.ceil(image.width * resize_factor)
        height = math.ceil(image.height * resize_factor)
        image = image.resize((width, height))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def load_resized_image(path: Path, settings: UitarsFormatSettings) -> Image.Image:
    image = load_pil_image(path)
    return resize_pil_image(image, max_pixels=settings.max_pixels, min_pixels=settings.min_pixels)


def image_content_item(image: Image.Image) -> dict[str, Any]:
    encoded = pil_to_base64(image)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{encoded}"},
    }


def staged_image_content_item(relative_path: str) -> dict[str, Any]:
    return {"type": "image", "path": relative_path}


def resolve_staged_image_item(item: dict[str, Any], rollout_dir: Path, settings: UitarsFormatSettings) -> dict[str, Any]:
    if item.get("type") != "image":
        return item
    relative_path = item.get("path")
    if not isinstance(relative_path, str):
        raise ValueError("Image content item must include a string `path`.")
    image = load_resized_image(rollout_dir / relative_path, settings)
    return image_content_item(image)


def resolve_staged_messages(
    messages: Sequence[dict[str, Any]],
    rollout_dir: Path,
    settings: UitarsFormatSettings,
) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, list):
            raise ValueError("Each message must have string role and list content.")
        resolved.append(
            {
                "role": role,
                "content": [
                    resolve_staged_image_item(item, rollout_dir, settings) if isinstance(item, dict) else item
                    for item in content
                ],
            }
        )
    return resolved


def build_user_prompt(instruction: str, settings: UitarsFormatSettings) -> str:
    return resolve_prompt_template(settings).format(
        instruction=instruction,
        action_space=resolve_action_space(settings),
        language=settings.language,
    )


def normalize_assistant_response(response: str) -> str:
    return add_box_token(response)


def build_messages_from_images_and_responses(
    instruction: str,
    image_paths: Sequence[Path],
    history_responses: Sequence[str],
    settings: UitarsFormatSettings,
    *,
    target_response: str | None = None,
    stage_relative_paths: bool = False,
    rollout_dir: Path | None = None,
) -> list[dict[str, Any]]:
    if len(image_paths) != len(history_responses) + 1:
        raise ValueError("Expected one more image than history responses.")

    windowed_paths = list(image_paths)
    if len(windowed_paths) > settings.history_n:
        windowed_paths = windowed_paths[-settings.history_n :]
    offset = len(image_paths) - len(windowed_paths)

    images = [load_resized_image(path, settings) for path in windowed_paths]
    windowed_responses = list(history_responses)[offset:]

    user_prompt = build_user_prompt(instruction, settings)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]

    def append_image(path_index: int) -> None:
        if stage_relative_paths:
            if rollout_dir is None:
                raise ValueError("rollout_dir is required when stage_relative_paths is true.")
            relative = windowed_paths[path_index].relative_to(rollout_dir).as_posix()
            messages.append({"role": "user", "content": [staged_image_content_item(relative)]})
        else:
            messages.append({"role": "user", "content": [image_content_item(images[path_index])]})

    image_num = 0
    if windowed_responses:
        for history_idx, history_response in enumerate(windowed_responses):
            if history_idx + settings.history_n > len(windowed_responses):
                append_image(image_num)
                image_num += 1
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": normalize_assistant_response(history_response)}],
                }
            )
        append_image(image_num)
    else:
        append_image(image_num)

    if target_response is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": normalize_assistant_response(target_response)}],
            }
        )
    return messages
