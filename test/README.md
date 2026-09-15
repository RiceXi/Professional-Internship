# 测试与证据

## CPU 测试

```bash
bash scripts/setup_cpu.sh
.venv-cpu/bin/python test/run_cpu.py --coverage
```

覆盖页表/动态分配、连续空闲区合并、fork/COW、独立释放、LRU、满容量交换、固定工作集、后端失败回滚、随机生命周期、运行器和实验汇总。运行器失败不会被计为完成请求。

覆盖率范围为整个 `src/memory`，阈值 ≥70%，包括尚未执行的 GPU 适配代码。未安装 torch 时三项实际张量测试明确跳过；含 PyTorch 的 CPU 环境会核对真实张量的 COW、页往返和交换。它们都不能证明 CUDA 内核正确。

`experiments/local/test_summary.json` 保存通过/失败/跳过、环境与源码哈希；`coverage.json` 保存逐文件行覆盖率。`--output <目录>` 可以指定归档位置。

## 课程 GPU 验证

```bash
bash scripts/validate_gpu.sh --storage-only
bash scripts/validate_gpu.sh --model ~/huggingface/Qwen3-0.6B
python scripts/gpu_matrix.py --model ~/huggingface/Qwen3-0.6B
```

真实 CUDA 页内容测试先执行，再比较显存充足和换页/COW 路径的贪心输出。缺环境退出 2，失败退出 1，通过退出 0。`storage_only` 报告的通过只针对页内容测试。

矩阵在独立进程改变并发与上下文长度，保留标准输出、异常、逐案例报告和矩阵摘要。完整 Attention 工作集超限单列为容量边界；没有真实 CUDA OOM 时不得报告其降低比例。详见[复现与演示](../docs/04-复现与演示.md)。

## 推理支撑模块测试

```bash
python test/run_all.py
```

需要 CUDA 与本地 Qwen3-0.6B。覆盖原有引擎、采样、前缀缓存与混合调度。默认使用设备 0，遵守 `CUDA_VISIBLE_DEVICES`；这些测试尚未在本机运行。
