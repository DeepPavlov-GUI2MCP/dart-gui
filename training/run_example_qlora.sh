#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
python training/run_qlora.py --config training/configs/qlora_example.yml
