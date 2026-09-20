# 测试

## CPU 测试

```bash
bash scripts/setup_cpu.sh
.venv-cpu/bin/python test/run_cpu.py --coverage
```

测试包括页表映射、动态分配、连续空闲区合并、fork/COW、LRU 换页、工作集固定和资源回收。随机序列用例检查长时间运行后的状态一致性，失败用例检查复制异常和容量不足时的回滚。

覆盖率统计整个 `src/memory`，通过阈值为 70%。安装 PyTorch 后，同一测试命令还会检查 CPU 张量的 COW、页往返和交换；未安装时跳过这三项。CUDA 适配代码也计入覆盖范围，但需要在 GPU 环境另行测试。

`experiments/local/test_summary.json` 保存通过/失败/跳过、环境与源码哈希；`coverage.json` 保存逐文件行覆盖率。`--output <目录>` 可以指定归档位置。

## PowerShell 入口测试

```powershell
powershell -ExecutionPolicy Bypass -File test\powershell\test_python_env.ps1
```

覆盖 `scripts/python_env.ps1` 的解释器探测：显式 `-Python` 路径、`py` 启动器可用、`py` 启动器存在但 `py -3` 失败时回退到 `python`/`python3`、候选解释器无法启动时跳过，以及最终找不到解释器时的报错。用例把伪造的启动器放在 `PATH` 最前面，不改动本机 Python 安装；最后一项会在禁用解释器的环境中运行 `scripts/setup_cpu.ps1`，确认入口不再因 `NativeCommandError` 中断。该脚本不需要 Pester，在 Windows PowerShell 5.1 与 PowerShell 7 下均可运行。

## GPU 验证

```bash
bash scripts/validate_gpu.sh --storage-only
bash scripts/validate_gpu.sh --model ~/huggingface/Qwen3-0.6B
python scripts/gpu_matrix.py --model ~/huggingface/Qwen3-0.6B
```

脚本先检查 CUDA 页内容，再比较显存充足和换页/COW 路径的贪心输出。通过时退出 0，执行失败时退出 1，环境不完整时退出 2。`--storage-only` 只运行页内容测试。

压力测试在独立进程中改变并发数和上下文长度，保存每组输出、异常及汇总结果。完整 Attention 工作集超限和 CUDA OOM 分别记录，详见[复现与演示](../docs/04-复现与演示.md)。

## 引擎与算子测试

```bash
python test/run_all.py
```

测试引擎、采样、前缀缓存与混合调度，需要 CUDA 和本地 Qwen3-0.6B。默认使用设备 0，支持通过 `CUDA_VISIBLE_DEVICES` 选择设备。目前尚无这组测试的 GPU 运行结果。
