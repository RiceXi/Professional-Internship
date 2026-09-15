# KV Cache 分页内存管理

本项目关注大模型推理中的 KV Cache 内存管理，实现了按需分页、前缀共享、写时复制（COW）和 LRU 换页，并接入 Qwen3 推理。

生成长度事先很难确定：按最大长度预留 KV 空间容易浪费，多个分支分别保存相同前缀也会重复占用显存。本项目将上下文拆成固定大小的页，在生成过程中逐步分配；分支先共享已有页，写入时再复制；显存不足时，将暂时不用的页换到主机内存。

内存管理逻辑可以脱离 GPU 运行。CPU 后端用于检查页表、复制和换页行为，张量后端连接真实 KV 池。当前已完成 CPU 测试和仿真实验，CUDA/Qwen3 路径已接入，尚待 GPU 实测。

## 快速开始

CPU 仿真只需要 Python 3.10+：

```bash
python3 simulate.py --seed 42 --frames 32 --block-size 4
python3 scripts/demo_memory.py
```

第一个命令在相同负载下比较连续预分配、按需分页和分页换页，结果写入 `experiments/local/simulation.json`。第二个命令演示 fork、COW、LRU 换页和资源回收，可以直接观察父子序列的内容变化。

安装测试依赖并运行测试：

```bash
bash scripts/setup_cpu.sh
```

脚本会创建 `.venv-cpu` 并检查覆盖率。之后可单独运行：

```bash
.venv-cpu/bin/python test/run_cpu.py --coverage
```

测试结果保存在 `experiments/local/`。更多说明见 [test/README.md](test/README.md)。

## GPU 推理与验证

GPU 引擎面向 Linux + NVIDIA CUDA，建议使用 Python 3.12。先按 [PyTorch](https://pytorch.org/get-started/locally/)、[FlashAttention](https://github.com/Dao-AILab/flash-attention#installation-and-features) 和 [FlashInfer](https://docs.flashinfer.ai/installation.html) 的安装说明配置依赖，再准备项目环境和模型：

```bash
python -m pip install -r requirements.txt
python -m pip check
hf download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B
python run.py --model Qwen3-0.6B --temperature 0 --eager --no-compile
```

`run.py` 支持 `--batch` 批量生成、`--cg` Decode CUDA Graph 和 `--compile` Prefill 编译。配置优先级为命令行、`config.yaml`、代码默认值。

分页和换页使用单独的验证入口：

```bash
bash scripts/validate_gpu.sh --model ~/huggingface/Qwen3-0.6B
```

默认使用 4 个 GPU KV 块、16 个主机块和 3 条请求，对比换页前后的生成结果，并记录显存峰值和搬运开销。参数可通过 `--help` 查看，环境和压力测试说明见[复现与演示](docs/04-复现与演示.md)。

换页允许多个请求轮流使用显存，但单次 Attention 仍要求完整历史 KV 驻留。因此，它能扩展可保存的上下文总量，单条请求的长度仍受物理 KV 池限制。

## 实验与文档

仓库包含 129 组 CPU 仿真的原始数据、图表和实验脚本，覆盖页大小、并发数、上下文长度、共享分支及外部碎片。CPU 实验衡量空间分配与容量，GPU 吞吐和延迟需要另行实测。

- [系统设计](docs/00-系统设计.md)：页表、COW 和换页流程。
- [显存与指标](docs/01-显存与指标.md)：KV 容量计算和统计方式。
- [评测报告](docs/02-量化评测报告.md)：实验配置、结果与分析。
- [复现与演示](docs/04-复现与演示.md)：完整运行步骤。

## 目录

```text
src/memory/      # 分页管理、存储后端、连续分配基线
src/control/     # 推理引擎与调度器
src/data/        # GPU KV 池、批数据和前缀索引
src/compute/     # Qwen3、Attention、采样和图执行
test/            # CPU 与 GPU 测试
bench/           # 内存实验和推理性能基准
experiments/     # 原始数据与图表；local/ 存放本地运行结果
scripts/         # 环境配置、演示和 GPU 验证脚本
docs/            # 设计、评测与使用说明
```
