#!/usr/bin/env python3
"""ML Space pytorch2 entry for UI-TARS LoRA SFT."""

from __future__ import annotations

import os
import sys
from pathlib import Path

requested_python = os.environ.get("DART_SFT_PYTHON", "").strip()
if requested_python:
    if not Path(requested_python).is_file():
        print(f"error: DART_SFT_PYTHON does not exist: {requested_python}", file=sys.stderr)
        raise SystemExit(1)
    requested_prefix = Path(requested_python).parent.parent.resolve()
    if Path(sys.prefix).resolve() != requested_prefix:
        os.execv(requested_python, [requested_python, __file__, *sys.argv[1:]])

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_DIR = REPO_ROOT / "training"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
os.chdir(REPO_ROOT)

config = os.environ.get("DART_SFT_CONFIG", "").strip()
if not config:
    print("error: DART_SFT_CONFIG must be set", file=sys.stderr)
    raise SystemExit(1)

pretokenized_dir = os.environ.get("DART_SFT_PRETOKENIZED_DIR", "").strip()
argv = ["--config", config]
if pretokenized_dir:
    argv.extend(["--from-pretokenized-traces", pretokenized_dir])

from run_sft import main

raise SystemExit(main(argv))
