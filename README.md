# Professional Internship · KV Cache 分页内存管理

专业实习课程任务一：将操作系统的分页、逻辑地址映射、共享与写时复制、页面置换用于大模型 KV Cache。验收依据为[任务书](面向大模型推理优化的操作系统级任务设计_合并版.docx)第二部分和通用考核要求。

## 课程目标与进度

| 内容 | 状态 |
|---|---|
| 无 GPU 分页仿真、页表、动态扩页、分配与回收 | 已实现，CPU 测试覆盖 |
| 连续预分配基线、固定种子负载、逐步资源统计 | 已实现 |
| 序列 fork 与写时复制 | 已实现，完整页/未满页及随机分支隔离测试通过 |
| CPU 换出、换入与 LRU | 已实现有界主机存储、LRU、工作集固定和满容量交换；CPU 测试通过 |
| 真实 KV 数据与引擎验证 | 已接入课程内存运行器与张量后端；CUDA/Qwen3 实测待在用户 GPU 上执行 |
| 评测报告、设计图、答辩材料 | 随阶段完善 |

持续批处理与 GPU 算子作为运行支撑。此次验收聚焦内存管理，详见[课程实施计划](docs/11-任务一验收与补齐方案.md)。

## 本地启动与测试

Python 3.10+。CPU 仿真运行只需标准库，覆盖率检查使用独立虚拟环境：

```bash
bash scripts/setup_cpu.sh
.venv-cpu/bin/python test/run_cpu.py --coverage
python3 simulate.py --seed 42 --frames 32 --block-size 4
```

`--coverage` 在测试失败或覆盖率低于 70% 时返回非零退出码；报告写入 `experiments/local/coverage.json`。仿真原始数据默认写入 `experiments/local/simulation.json`，包括负载、每步指标、完成和容量拒绝记录。容量拒绝属于仿真结果，不是真实 CUDA OOM；仿真 tick 不是 GPU 时间。

### RTX 4060 Laptop 验证入口

在具备兼容依赖的 Linux CUDA 环境中运行：

```bash
bash scripts/validate_gpu.sh --model ~/huggingface/Qwen3-0.6B
```

默认固定 4 个 GPU KV 块、16 个主机块、3 条请求，所有参数可在 `--help` 中调整。显存充足的参考配置仅用于输出正确性检查；性能与容量对照使用同样的物理块数。GPU 报告写入 `experiments/local/gpu_validation.json`。未安装 CUDA 依赖或无 GPU 时退出码为 2，并标记 unavailable。

## 仓库结构

```text
src/memory/      # CPU 安全的内存管理逻辑、存储后端与连续基线
src/control/     # 推理引擎与调度器
src/data/        # 现有 GPU KV 池、批数据和前缀索引
src/compute/     # Qwen3、Attention、采样和图执行
simulate.py      # 无 GPU 课程实验入口
run.py           # 现有 GPU 推理入口
test/cpu/        # 课程逻辑测试
test/            # GPU 功能测试及测试运行器
bench/           # GPU 性能基准
experiments/     # 实验数据；local/ 为本地临时结果
scripts/         # 环境与复现脚本
docs/            # 课程设计、指标与验收文档
```

## GPU 推理环境

现有引擎面向 Linux + NVIDIA CUDA，建议 Python 3.12。按 [PyTorch](https://pytorch.org/get-started/locally/)、[FlashAttention](https://github.com/Dao-AILab/flash-attention#installation-and-features) 和 [FlashInfer](https://docs.flashinfer.ai/installation.html) 官方说明安装相互兼容且支持目标 GPU 的版本，再安装 `requirements.txt`。CPU 仿真环境无需这些依赖。

```bash
python -m pip install -r requirements.txt
python -m pip check
hf download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B
python run.py --model Qwen3-0.6B --temperature 0 --eager --no-compile
python test/run_all.py
```

可用 `--cg` 启用 Decode CUDA Graph，`--compile` 启用 Prefill 编译，`--batch` 运行批量演示。配置优先级为命令行、`config.yaml`、代码默认值。内存管理实验应固定执行配置，分别报告模型、KV、运行时显存和数据搬运代价。

现有 GPU 测试要求 CUDA 和本地模型，不能由 CPU 仿真替代。GPU 性能脚本见 [bench/README.md](bench/README.md)，课程设计文档见 [docs/README.md](docs/README.md)。

## 交付与回滚

每阶段通过相关测试后单独提交和推送。源码、可复现命令与原始实验数据共同构成验收证据。例会、互审和个人分工按实际过程记录；历史性能数字不作为本次课程结论。
