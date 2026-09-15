"""清空 bench/out 后按序跑 v3 性能用例，结果只写 txt。

默认只跑本仓库的 v3；传入 --with-external 后追加 nano-vllm 和官方 vLLM。
默认使用设备 0，尊重已有 CUDA_VISIBLE_DEVICES 设置。

用法（在仓库根目录下）:
    python bench/run_all.py
    VLLM_PYTHON=/path/to/vllm-env/bin/python python bench/run_all.py --with-external
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
REPO = ROOT.parent
PY = Path(sys.executable)
OUT = BENCH / "out"


def run(py: Path, *args: str, env: dict[str, str]) -> None:
    cmd = [str(py), *args]
    print(f"\n{'=' * 64}\n$ {' '.join(cmd)}\n{'=' * 64}\n", flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-external",
        action="store_true",
        help="追加同级 nano-vllm 与独立官方 vLLM 环境的对比",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    print(
        f"GPU CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
        f"(PCI_BUS_ID)",
        flush=True,
    )

    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)

    for name in (
        "bench_prefill.py",
        "bench_e2e.py",
    ):
        run(PY, f"bench/{name}", env=env)

    run(PY, "bench/bench_compare.py", "--engine", "v3", env=env)
    run(PY, "bench/bench_ttft_decode.py", "--engine", "v3", env=env)
    run(PY, "bench/bench_concurrent.py", "--engine", "v3", env=env)

    if args.with_external:
        nano_root = REPO / "nano-vllm"
        if not nano_root.is_dir():
            sys.exit(f"missing sibling repository: {nano_root}")
        vllm_python = os.environ.get("VLLM_PYTHON")
        if not vllm_python or not Path(vllm_python).is_file():
            sys.exit("set VLLM_PYTHON to the Python executable of the vLLM environment")
        pyv = Path(vllm_python)
        run(PY, "bench/bench_compare.py", "--engine", "nano", env=env)
        run(pyv, "bench/bench_compare.py", "--engine", "vllm", env=env)
        run(PY, "bench/bench_ttft_decode.py", "--engine", "nano", env=env)
        run(pyv, "bench/bench_ttft_decode.py", "--engine", "vllm", env=env)
        run(PY, "bench/bench_concurrent.py", "--engine", "nano", env=env)
        run(pyv, "bench/bench_concurrent.py", "--engine", "vllm", env=env)

    print("\n======== bench/out ========")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name:24s} {p.stat().st_size:6d} B")


if __name__ == "__main__":
    main()
