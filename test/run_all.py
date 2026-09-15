"""按序运行 test 目录下全部功能正确性用例，任一失败即中止。

默认使用设备 0，尊重已有 CUDA_VISIBLE_DEVICES 设置。

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


def main() -> None:
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

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
