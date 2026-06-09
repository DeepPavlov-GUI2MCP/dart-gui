#!/usr/bin/env python3
"""Submit dart-gui ML Space jobs via client_lib."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jobs.hf_cache import forward_hf_token_env, hf_hub_env

import client_lib

REPO_ROOT = _REPO_ROOT

REGION = os.environ.get("DART_JOB_REGION", "A100-MT")
USR_NAME = os.environ.get("DART_USR_NAME", "username")
PRJ_TAGS = os.environ.get("DART_PRJ_TAGS", "#ID0045 #rnd")

N_GPUS_TO_INSTANCE_TYPE = {
    1: "a100.1gpu.8C.243G",
    2: "a100.2gpu.16C.486G",
    4: "a100.4gpu.32C.972G",
    8: "a100.8gpu.64C.1944G",
}

DEFAULT_QUEUE_NAME = os.environ.get("DART_JOB_QUEUE_NAME", "rnd-gigachat-embs").strip()


def repo_path(*parts: str) -> str:
    return str(REPO_ROOT.joinpath(*parts))


def queue_name_from_env() -> str | None:
    raw = os.environ.get("DART_JOB_QUEUE_NAME", DEFAULT_QUEUE_NAME).strip()
    return raw or None


_PRIORITY_BY_NAME = {
    "low": client_lib.PriorityClass.low,
    "medium": client_lib.PriorityClass.medium,
    "high": client_lib.PriorityClass.high,
}


def priority_class_from_env() -> client_lib.PriorityClass | None:
    raw = os.environ.get("DART_JOB_PRIORITY_CLASS", "high").strip().lower()
    if not raw or raw in ("none", "default"):
        return None
    if raw not in _PRIORITY_BY_NAME:
        supported = ", ".join(sorted(_PRIORITY_BY_NAME))
        raise ValueError(
            f"DART_JOB_PRIORITY_CLASS={raw!r} is not supported. "
            f"Choose one of: {supported} (or none to omit)"
        )
    return _PRIORITY_BY_NAME[raw]


def resolve_job_image(*, default_image: str | None = None) -> str:
    image = os.environ.get("DART_JOB_IMAGE", "").strip()
    if image:
        return image
    if default_image:
        return default_image
    raise ValueError(
        "Set DART_JOB_IMAGE to the full registry URL of a pushed job image "
        "(see jobs/README.md)."
    )


def require_sft_config() -> str:
    config = os.environ.get("DART_SFT_CONFIG", "").strip()
    if not config:
        raise ValueError("DART_SFT_CONFIG must point to a train_sft YAML config.")
    return config


def resolve_sft_run_id() -> str:
    run_id = os.environ.get("DART_SFT_RUN_ID", "").strip()
    if not run_id:
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        os.environ["DART_SFT_RUN_ID"] = run_id
    return run_id


def base_job_kwargs(*, script_rel: str, default_image: str | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "job_desc": f"{USR_NAME} | {PRJ_TAGS}",
        "base_image": resolve_job_image(default_image=default_image),
        "script": repo_path(script_rel),
        "region": REGION,
        "preflight_check": False,
        "flags": {},
    }
    priority = priority_class_from_env()
    if priority is not None:
        kwargs["priority_class"] = priority
    return kwargs


def build_sft_env() -> dict[str, str]:
    num_nodes = max(1, int(os.environ.get("DART_JOB_NUM_NODES", "1")))
    gpus_per_node = max(1, int(os.environ.get("DART_JOB_GPUS_PER_NODE", "1")))
    env_variables: dict[str, str] = {
        "PYTHONPATH": repo_path(),
        "DART_SFT_CONFIG": require_sft_config(),
        "DART_SFT_RUN_ID": resolve_sft_run_id(),
        "DART_SFT_SAVE_RANK_LOGS": os.environ.get("DART_SFT_SAVE_RANK_LOGS", "1"),
        "DART_SFT_TIMESTAMPED_OUTPUT": os.environ.get("DART_SFT_TIMESTAMPED_OUTPUT", "1"),
        "NCCL_DEBUG": os.environ.get("DART_NCCL_DEBUG", "WARN"),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
            "DART_PYTORCH_CUDA_ALLOC_CONF",
            "expandable_segments:True",
        ),
        "DART_JOB_NUM_NODES": str(num_nodes),
        "DART_JOB_GPUS_PER_NODE": str(gpus_per_node),
        "DART_SFT_PYTHON": os.environ.get(
            "DART_SFT_PYTHON",
            "/opt/dart-sft-venv/bin/python3",
        ),
    }
    pretokenized_dir = os.environ.get("DART_SFT_PRETOKENIZED_DIR", "").strip()
    if pretokenized_dir:
        env_variables["DART_SFT_PRETOKENIZED_DIR"] = pretokenized_dir
    run_root = os.environ.get("DART_SFT_RUN_ROOT", "").strip()
    if run_root:
        env_variables["DART_SFT_RUN_ROOT"] = run_root
    env_variables.update(hf_hub_env())
    env_variables.update(forward_hf_token_env())
    for key in ("MLFLOW_HOST", "MLFLOW_PORT", "MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME"):
        value = os.environ.get(key, "").strip()
        if value:
            env_variables[key] = value
    return env_variables


def build_sft_job_kwargs() -> dict[str, Any]:
    num_nodes = max(1, int(os.environ.get("DART_JOB_NUM_NODES", "1")))
    gpus_per_node = max(1, int(os.environ.get("DART_JOB_GPUS_PER_NODE", "1")))
    if gpus_per_node not in N_GPUS_TO_INSTANCE_TYPE:
        supported = ", ".join(str(n) for n in sorted(N_GPUS_TO_INSTANCE_TYPE))
        raise ValueError(
            f"DART_JOB_GPUS_PER_NODE={gpus_per_node} is not supported. "
            f"Choose one of: {supported}"
        )
    kwargs = base_job_kwargs(script_rel="jobs/sft/run_uitars_lora/run.py")
    kwargs.update(
        {
            "n_workers": num_nodes,
            "processes_per_worker": gpus_per_node,
            "instance_type": N_GPUS_TO_INSTANCE_TYPE[gpus_per_node],
            "type": "pytorch2",
            "pytorch_use_env": True,
            "env_variables": build_sft_env(),
        }
    )
    queue = queue_name_from_env()
    if queue:
        kwargs["queue_name"] = queue
    return kwargs


@dataclass(frozen=True)
class JobDefinition:
    name: str
    description: str
    job_dir: str
    build_kwargs: Callable[[], dict[str, Any]]


JOB_REGISTRY: dict[str, JobDefinition] = {
    "sft": JobDefinition(
        name="sft",
        description="UI-TARS LoRA SFT from pretokenized shards (pytorch2 GPU job)",
        job_dir="sft/run_uitars_lora",
        build_kwargs=build_sft_job_kwargs,
    ),
}


def presubmit_env_for(definition: JobDefinition) -> dict[str, str]:
    env = os.environ.copy()
    env.update(hf_hub_env())
    if definition.name == "sft":
        env.update(build_sft_env())
    return env


def run_presubmit(definition: JobDefinition) -> None:
    presubmit = REPO_ROOT / "jobs" / definition.job_dir / "presubmit.py"
    if not presubmit.is_file():
        return
    print(f"Running presubmit: {presubmit.relative_to(REPO_ROOT)}")
    subprocess.run(
        [sys.executable, str(presubmit)],
        cwd=REPO_ROOT,
        env=presubmit_env_for(definition),
        check=True,
    )


def submit_job(job_name: str) -> str:
    key = job_name.strip().lower()
    if key not in JOB_REGISTRY:
        known = ", ".join(sorted(JOB_REGISTRY))
        raise ValueError(f"Unknown job {job_name!r}. Choose one of: {known}")
    definition = JOB_REGISTRY[key]
    run_presubmit(definition)
    job = client_lib.Job(**definition.build_kwargs())
    return job.submit()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Submit a dart-gui job to ML Space via client_lib.")
    parser.add_argument(
        "job",
        nargs="?",
        default=None,
        help="Job name (sft). Defaults to DART_JOB env var.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List registered job names and exit.",
    )
    parser.add_argument(
        "--priority",
        choices=sorted(_PRIORITY_BY_NAME),
        default=None,
        help="Queue priority (default: high, or DART_JOB_PRIORITY_CLASS).",
    )
    args = parser.parse_args(argv)

    if args.priority is not None:
        os.environ["DART_JOB_PRIORITY_CLASS"] = args.priority

    if args.list:
        for name, definition in sorted(JOB_REGISTRY.items()):
            print(f"{name}: {definition.description}")
        return 0

    job_name = args.job
    if job_name is None:
        job_name = os.environ.get("DART_JOB", "").strip()
    if not job_name:
        parser.error("Provide a job name (sft) or set DART_JOB. Use --list to see options.")

    print(submit_job(job_name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
