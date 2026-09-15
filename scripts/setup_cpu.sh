#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON:-python3}" -m venv .venv-cpu
.venv-cpu/bin/python -m pip install -r requirements-cpu.lock
.venv-cpu/bin/python test/run_cpu.py --coverage
