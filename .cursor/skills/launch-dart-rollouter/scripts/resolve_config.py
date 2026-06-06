#!/usr/bin/env python3
"""Resolve device + profile from manifest.yaml into launch settings."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


SCRIPT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = Path(
    os.environ.get(
        "DART_ROLLOUTER_MANIFEST",
        SCRIPT_DIR.parent / "config" / "manifest.yaml",
    )
)


def _infer_repo_root(manifest_path: Path) -> str | None:
    """Infer repo root from .../repo/.cursor/skills/launch-dart-rollouter/config/manifest.yaml."""
    try:
        candidate = manifest_path.resolve().parents[4]
    except IndexError:
        return None
    if (candidate / "validation" / "model_service.py").is_file():
        return str(candidate)
    return None


def _resolve_repo_root(device: dict[str, Any], manifest_path: Path) -> str:
    if os.environ.get("REPO_ROOT"):
        return os.environ["REPO_ROOT"]
    inferred = _infer_repo_root(manifest_path)
    if inferred:
        return inferred
    return device.get("repo_root") or "/workspace/dart-gui"


def _resolve_python(device: dict[str, Any], repo_root: str) -> str:
    python = device.get("python") or ".venv/bin/python"
    path = Path(python)
    if not path.is_absolute():
        path = Path(repo_root) / path
    return str(path)


def _load_manifest(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text)
    else:
        raise RuntimeError(
            "PyYAML is required to read manifest.yaml (pip install pyyaml)"
        )
    if not isinstance(data, dict):
        raise ValueError(f"Invalid manifest: {path}")
    return data


def _merge_requirements(device: dict[str, Any], profile: dict[str, Any]) -> str:
    return profile.get("requirements") or device.get("requirements", "")


def resolve(
    manifest: dict[str, Any],
    device_name: str | None,
    profile_name: str | None,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, Any]:
    devices = manifest.get("devices") or {}
    profiles = manifest.get("profiles") or {}

    device_name = device_name or manifest.get("default_device") or next(iter(devices), "")
    profile_name = profile_name or manifest.get("default_profile") or next(iter(profiles), "")

    if device_name not in devices:
        raise KeyError(
            f"Unknown device {device_name!r}. Known: {', '.join(sorted(devices))}"
        )
    if profile_name not in profiles:
        raise KeyError(
            f"Unknown profile {profile_name!r}. Known: {', '.join(sorted(profiles))}"
        )

    device = devices[device_name]
    profile = profiles[profile_name]
    vllm = dict(profile.get("vllm") or {})
    tp = int(vllm.get("tensor_parallel_size", 1))

    repo_root = _resolve_repo_root(device, manifest_path)
    python = _resolve_python(device, repo_root)
    venv_mode = device.get("venv_mode", "complementary")
    vllm_port = int(device.get("vllm_port", 8010))
    model_port = int(device.get("model_port", 15961))

    hydra_overrides = [
        f"model.ckpt_path={profile['ckpt_path']}",
        f"model.replicas={profile.get('replicas', 1)}",
        f"model.base_port={vllm_port}",
        "model.host=0.0.0.0",
        f"model.service_port={model_port}",
        f"model.service_endpoint=http://localhost:{model_port}",
    ]
    for key, value in vllm.items():
        hydra_overrides.append(f"++model.vllm_params.{key}={value}")

    service_cmd = (
        f"source {repo_root}/.venv/bin/activate && "
        f"cd {repo_root}/validation && "
        f"python model_service.py --config-name {profile.get('hydra_config', 'config_singleapp')} "
        + " ".join(hydra_overrides)
    )

    return {
        "device": device_name,
        "profile": profile_name,
        "description": profile.get("description", ""),
        "ssh_host": device.get("ssh_host"),
        "repo_root": repo_root,
        "session_name": device.get("session_name", f"dart-rollouter-{device_name}"),
        "python": python,
        "venv_mode": venv_mode,
        "requirements": _merge_requirements(device, profile),
        "vllm_port": vllm_port,
        "model_port": model_port,
        "tensor_parallel_size": tp,
        "service_cmd": service_cmd,
    }


def _emit_shell(resolved: dict[str, Any]) -> None:
    for key, value in resolved.items():
        if value is None:
            continue
        escaped = str(value).replace("'", "'\"'\"'")
        print(f"{key.upper()}='{escaped}'")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", "-d", default=os.environ.get("DEVICE"))
    parser.add_argument("--profile", "-p", default=os.environ.get("PROFILE"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help=f"Manifest path (default: {MANIFEST_PATH})",
    )
    parser.add_argument(
        "--format",
        choices=("shell", "json", "cmd"),
        default="shell",
        help="Output format",
    )
    parser.add_argument("--list", action="store_true", help="List devices and profiles")
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)

    if args.list:
        inferred = _infer_repo_root(args.manifest.resolve())
        if inferred:
            print(f"Detected repo_root: {inferred}")
        print("Devices:")
        for name, cfg in (manifest.get("devices") or {}).items():
            host = cfg.get("ssh_host") or "(local)"
            mode = cfg.get("venv_mode", "complementary")
            print(f"  {name}: ssh={host} session={cfg.get('session_name')} venv={mode}")
        print("Profiles:")
        for name, cfg in (manifest.get("profiles") or {}).items():
            tp = (cfg.get("vllm") or {}).get("tensor_parallel_size", 1)
            print(
                f"  {name}: {cfg.get('ckpt_path')} "
                f"replicas={cfg.get('replicas', 1)} tp={tp}"
            )
        return 0

    resolved = resolve(manifest, args.device, args.profile, args.manifest.resolve())

    if args.format == "json":
        print(json.dumps(resolved, indent=2))
    elif args.format == "cmd":
        print(resolved["service_cmd"])
    else:
        _emit_shell(resolved)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
