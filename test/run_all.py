"""按序运行 test 目录下全部功能正确性用例，任一失败即中止。

默认钉在本机 NVIDIA GeForce RTX 4090 D（nvidia-smi PCI 序）。

用法（在仓库根目录下）:
    python test/run_all.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = Path(sys.executable)

TESTS = (
    "test_sampler.py",
    "test_engine.py",
    "test_prefix.py",
    "test_scheduler.py",
)


def pick_4090() -> str:
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
        text=True,
    )
    for line in raw.strip().splitlines():
        idx, name = [x.strip() for x in line.split(",", 1)]
        if "4090" in name:
            return idx
    return "0"


def main() -> None:
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = pick_4090()

    failed = 0
    for name in TESTS:
        print(f"\n{'=' * 64}\n$ {PY} test/{name}\n{'=' * 64}\n", flush=True)
        r = subprocess.run([str(PY), f"test/{name}"], cwd=str(ROOT), env=env)
        if r.returncode != 0:
            failed += 1
            print(f"FAILED: {name}", flush=True)
            break

    if failed:
        sys.exit(1)
    print("\nall tests passed")


if __name__ == "__main__":
    main()
