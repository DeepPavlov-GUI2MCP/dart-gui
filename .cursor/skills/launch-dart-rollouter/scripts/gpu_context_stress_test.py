#!/usr/bin/env python3
"""Multimodal GPU stress test: fill context with real images and sample GPU stats.

Unlike text-only repetition, this sends repeated screenshot payloads (OpenAI
multimodal chat format) to exercise Holo3 vision encoding + KV cache the way
GUI rollouts do.

Example:
  python gpu_context_stress_test.py --max-model-len 65536 --context-image-pct 1.0
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import pynvml
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = SCRIPT_DIR.parent / "assets" / "context_sample.jpg"


@dataclass
class GpuSample:
    ts: float
    gpu_id: int
    mem_used_mib: int
    mem_total_mib: int
    sm_util_pct: int
    mem_util_pct: int


@dataclass
class RunStats:
    samples: List[GpuSample] = field(default_factory=list)
    request_started: Optional[float] = None
    request_ended: Optional[float] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    num_images: int = 0
    tokens_per_image: float = 0.0
    context_image_pct: float = 1.0
    error: Optional[str] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8010")
    p.add_argument("--model", default="Hcompany/Holo3-35B-A3B")
    p.add_argument("--max-model-len", type=int, default=65536)
    p.add_argument(
        "--context-image-pct",
        type=float,
        default=1.0,
        help="Fraction of prompt budget (0-1) to fill with images.",
    )
    p.add_argument(
        "--context-image",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"Screenshot to repeat (default: {DEFAULT_IMAGE})",
    )
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--sample-interval-ms", type=int, default=200)
    p.add_argument("--timeout-s", type=int, default=900)
    p.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Hard cap on images (0 = auto from context budget).",
    )
    return p.parse_args()


def image_data_url(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Context image not found: {path}")
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_image_messages(num_images: int, image_url: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for i in range(num_images):
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"GUI rollout screenshot step {i + 1}. "
                            "Describe visible widgets and state briefly."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        )
    return messages


def probe_prompt_tokens(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout_s: int,
) -> tuple[Optional[int], Optional[str]]:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 1,
        "temperature": 0,
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout_s)
    except Exception as exc:
        return None, str(exc)

    if resp.status_code == 400:
        return None, f"HTTP 400: {resp.text[:300]}"
    if not resp.ok:
        return None, f"HTTP {resp.status_code}: {resp.text[:300]}"

    usage = resp.json().get("usage") or {}
    return int(usage.get("prompt_tokens", 0)), None


def calibrate_tokens_per_image(
    base_url: str,
    model: str,
    image_url: str,
    timeout_s: int,
) -> float:
    one = build_image_messages(1, image_url)
    two = build_image_messages(2, image_url)
    t1, err1 = probe_prompt_tokens(base_url, model, one, timeout_s)
    t2, err2 = probe_prompt_tokens(base_url, model, two, timeout_s)
    if err1 or t1 is None:
        raise RuntimeError(f"Calibration failed (1 image): {err1}")
    if err2 or t2 is None:
        raise RuntimeError(f"Calibration failed (2 images): {err2}")
    delta = t2 - t1
    if delta <= 0:
        raise RuntimeError(f"Unexpected calibration delta: {t1} -> {t2}")
    return float(delta)


def find_image_count(
    base_url: str,
    model: str,
    image_url: str,
    target_prompt_tokens: int,
    tokens_per_image: float,
    timeout_s: int,
    max_images_cap: int,
) -> tuple[int, int]:
    """Find the largest image count that fits under target_prompt_tokens."""
    estimate = max(1, int(target_prompt_tokens / tokens_per_image * 0.995))
    hi = min(max_images_cap, estimate + 15) if max_images_cap else estimate + 15
    lo = max(1, estimate - 15)
    best_count = 0
    best_tokens = 0

    while lo <= hi:
        mid = (lo + hi) // 2
        tokens, err = probe_prompt_tokens(
            base_url, model, build_image_messages(mid, image_url), timeout_s
        )
        if err:
            hi = mid - 1
            continue
        assert tokens is not None
        if tokens <= target_prompt_tokens:
            if tokens > best_tokens:
                best_count = mid
                best_tokens = tokens
            lo = mid + 1
        else:
            hi = mid - 1

    if best_count == 0:
        raise RuntimeError(
            "Could not fit images under target prompt budget; "
            f"target={target_prompt_tokens}, tried up to {hi + 15}"
        )
    return best_count, best_tokens


class GpuMonitor:
    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self.stats = RunStats()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        pynvml.nvmlInit()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            ts = time.time()
            for gpu_id in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.stats.samples.append(
                    GpuSample(
                        ts=ts,
                        gpu_id=gpu_id,
                        mem_used_mib=mem.used // (1024 * 1024),
                        mem_total_mib=mem.total // (1024 * 1024),
                        sm_util_pct=util.gpu,
                        mem_util_pct=util.memory,
                    )
                )
            time.sleep(self.interval_s)


def summarize(stats: RunStats, max_model_len: int) -> dict:
    by_gpu: dict[int, list[GpuSample]] = {}
    for s in stats.samples:
        by_gpu.setdefault(s.gpu_id, []).append(s)

    def phase_samples(start: Optional[float], end: Optional[float]) -> List[GpuSample]:
        if start is None or end is None:
            return stats.samples
        return [s for s in stats.samples if start <= s.ts <= end]

    active = phase_samples(stats.request_started, stats.request_ended)

    gpu_summary = {}
    for gpu_id, samples in by_gpu.items():
        active_gpu = [s for s in active if s.gpu_id == gpu_id] or samples
        mem_used = [s.mem_used_mib for s in active_gpu]
        sm_util = [s.sm_util_pct for s in active_gpu]
        mem_total = samples[-1].mem_total_mib if samples else 0
        peak_mem = max(mem_used) if mem_used else 0
        gpu_summary[gpu_id] = {
            "peak_mem_mib": peak_mem,
            "avg_mem_mib": round(statistics.mean(mem_used), 1) if mem_used else 0,
            "headroom_mib": mem_total - peak_mem,
            "headroom_pct": round(100 * (mem_total - peak_mem) / mem_total, 1) if mem_total else 0,
            "peak_sm_util_pct": max(sm_util) if sm_util else 0,
            "avg_sm_util_pct": round(statistics.mean(sm_util), 1) if sm_util else 0,
            "mem_total_mib": mem_total,
        }

    total_context = stats.prompt_tokens + stats.completion_tokens
    return {
        "mode": "multimodal_images",
        "num_images": stats.num_images,
        "tokens_per_image_calibrated": round(stats.tokens_per_image, 1),
        "context_image_pct": stats.context_image_pct,
        "prompt_tokens": stats.prompt_tokens,
        "completion_tokens": stats.completion_tokens,
        "total_tokens": stats.total_tokens,
        "context_used_vs_limit": f"{total_context}/{max_model_len}",
        "context_fill_pct": round(100 * total_context / max_model_len, 1) if max_model_len else 0,
        "request_duration_s": round((stats.request_ended or 0) - (stats.request_started or 0), 2)
        if stats.request_started and stats.request_ended
        else None,
        "gpus": gpu_summary,
        "error": stats.error,
    }


def main() -> int:
    args = parse_args()
    if not 0 < args.context_image_pct <= 1.0:
        print("context-image-pct must be in (0, 1]", file=sys.stderr)
        return 1

    reserve = args.max_tokens + 128
    prompt_budget = args.max_model_len - reserve
    target_prompt_tokens = int(prompt_budget * args.context_image_pct)
    print(f"Image: {args.context_image}")
    print(
        f"Prompt budget: {target_prompt_tokens} tokens "
        f"({args.context_image_pct:.0%} of {prompt_budget}, limit={args.max_model_len})"
    )

    image_url = image_data_url(args.context_image)
    print("Calibrating tokens/image with live vLLM probes...")
    tokens_per_image = calibrate_tokens_per_image(
        args.base_url, args.model, image_url, min(args.timeout_s, 120)
    )
    print(f"Calibrated ~{tokens_per_image:.1f} prompt tokens per image")

    est_images = max(1, int(target_prompt_tokens / tokens_per_image))
    max_cap = args.max_images or max(est_images + 20, 4)
    print(f"Searching image count (estimate ~{est_images}, cap {max_cap})...")
    num_images, probe_tokens = find_image_count(
        args.base_url,
        args.model,
        image_url,
        target_prompt_tokens,
        tokens_per_image,
        min(args.timeout_s, 300),
        max_cap,
    )
    print(f"Selected {num_images} images (~{probe_tokens} prompt tokens from probe)")

    messages = build_image_messages(num_images, image_url)
    monitor = GpuMonitor(args.sample_interval_ms / 1000)
    monitor.stats.num_images = num_images
    monitor.stats.tokens_per_image = tokens_per_image
    monitor.stats.context_image_pct = args.context_image_pct
    monitor.start()
    time.sleep(0.5)

    url = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": 0,
    }

    print(f"POST {url} with {num_images} images, max_tokens={args.max_tokens} ...")
    monitor.stats.request_started = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=args.timeout_s)
        monitor.stats.request_ended = time.time()
        if not resp.ok:
            monitor.stats.error = f"HTTP {resp.status_code}: {resp.text[:800]}"
        else:
            data = resp.json()
            usage = data.get("usage") or {}
            monitor.stats.prompt_tokens = int(usage.get("prompt_tokens", probe_tokens))
            monitor.stats.completion_tokens = int(usage.get("completion_tokens", 0))
            monitor.stats.total_tokens = int(usage.get("total_tokens", 0))
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"OK — completion preview: {content[:120]!r}...")
    except Exception as exc:
        monitor.stats.request_ended = time.time()
        monitor.stats.error = str(exc)

    monitor.stop()
    report = summarize(monitor.stats, args.max_model_len)
    print("\n=== GPU multimodal context stress report ===")
    print(json.dumps(report, indent=2))

    if report.get("error"):
        return 1

    gpus = report["gpus"]
    min_headroom = min(g["headroom_mib"] for g in gpus.values()) if gpus else 0
    max_sm = max(g["peak_sm_util_pct"] for g in gpus.values()) if gpus else 0
    print("\n=== Interpretation ===")
    print(
        f"Filled context with {report['num_images']} real images "
        f"({report['tokens_per_image_calibrated']} tok/image calibrated)."
    )
    if min_headroom > 4096:
        print(f"Memory headroom: ~{min_headroom/1024:.1f} GB free on the tightest GPU.")
    elif min_headroom > 1024:
        print(f"Memory headroom: ~{min_headroom/1024:.1f} GB free — modest room remaining.")
    else:
        print(f"Memory headroom: ~{min_headroom} MiB free — near capacity under multimodal load.")

    if max_sm < 50:
        print(f"Compute headroom: peak SM util {max_sm}% — memory-bound.")
    else:
        print(f"Compute headroom: peak SM util {max_sm}% — significant compute during vision+prefill.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
