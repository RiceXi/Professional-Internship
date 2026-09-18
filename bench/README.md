# 基准测试

## 内存分配实验

```bash
python3 bench/bench_memory.py
```

共 129 组确定性 CPU 仿真，对照连续预分配、按需分页、分页与主机换页；另有共享分支 COW 和外部碎片用例。记录页大小、物理容量、负载、每步资源状态、完成与容量拒绝，以及源码哈希。

实验脚本输出 `experiments/task1/raw.json` 和 `summary.csv`，本次评测的 PNG/SVG 图表也保存在该目录。策略共用负载和轮询规则，换页额外使用有界主机页池；CPU tick 不是设备时间，容量拒绝不是 CUDA OOM。

GPU 对照运行 `python scripts/gpu_matrix.py`。详见[量化报告](../docs/02-量化评测报告.md)和[复现说明](../docs/04-复现与演示.md)。

## 真实轨迹实验

```bash
python3 bench/bench_trace.py --trace <trace.csv> --format azure \
  --limit 300 --block-size 16 --frames 128 --host-pages 128 \
  --output experiments/task1/trace --export-lengths experiments/task1/trace/azure-sample-lengths.json
```

前面的内存实验使用合成负载（长度 3～20 token）。本脚本改用真实推理轨迹或对话数据集，输出与合成实验相同的汇总列，便于并列比较。

| `--format` | 输入 | 到达时间 |
|---|---|---|
| `azure` | Azure LLM 推理轨迹 CSV（`TIMESTAMP,ContextTokens,GeneratedTokens`） | 真实时间戳按 `--tick-seconds` 换算 |
| `sharegpt` | ShareGPT 格式 JSON 对话 | 泊松流，`--arrival-rate` 与 `--seed` 控制 |
| `longbench` | LongBench jsonl（`context`、`input`） | 泊松流；输出长度用 `--output-tokens` 假定 |
| `lengths` | 本脚本导出的长度分布（不含原文） | 记录中的 tick 原样重放 |

建模约定：一个 tick 对应一次解码迭代，`--tick-seconds`（默认 0.05 s）把墙钟时间换算成 tick；`--max-tokens` 是连续基线为每条请求预留的 `max_model_len`，超长请求按拒绝处理并计数，不做静默截断。`longbench` 与 `sharegpt` 的文本长度用字符数估算，需要精确值时在 `bench/workloads.py` 中传入自己的分词器。

仓库归档的 Azure 轨迹结果位于 `experiments/task1/trace/`。归档时只保存推导出的长度分布，不含对话原文，重放命令：

```bash
python3 bench/bench_trace.py --trace experiments/task1/trace/azure-sample-lengths.json --format lengths
```

该命令约需 10 秒，结果与归档的 `summary.csv` 逐列一致。测试默认只校验连续预分配与按需分页两行，设置 `KV_TRACE_SLOW_REPLAY=1` 后连同换页行一起校验。

## 推理性能

推理性能测试需要 CUDA 和本地模型。默认使用设备 0，也可以通过 `CUDA_VISIBLE_DEVICES` 选择设备。

```bash
python bench/run_all.py
```

| 脚本 | 指标或对照 |
|---|---|
| `bench_prefill.py` | Prefill 首 token 耗时与编译配置 |
| `bench_e2e.py` | 连续批吞吐、CUDA Graph 和编译配置 |
| `bench_compare.py` | 端到端跨引擎吞吐，可选外部引擎 |
| `bench_ttft_decode.py` | TTFT、TPOT 与前缀命中 |
| `bench_concurrent.py` | 高并发吞吐 |

结果写入被 Git 忽略的 `bench/out/`；批量运行前会清空此目录，需归档的结果应先另存。单项命令中的 `--engine v3` 是保留的本项目引擎标识，不要求安装其他版本。默认运行不需要外部引擎；部分辅助微基准引用外部实现路径，运行前查看其依赖。

这组测试比较计算执行配置；分页、COW 和换页的对照使用前面的内存实验脚本。
