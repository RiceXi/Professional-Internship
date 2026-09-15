# 基准测试

## 内存分配实验

```bash
python3 bench/bench_memory.py
```

共 129 组确定性 CPU 仿真，对照连续预分配、按需分页、分页与主机换页；另有共享分支 COW 和外部碎片用例。记录页大小、物理容量、负载、每步资源状态、完成与容量拒绝，以及源码哈希。

实验脚本输出 `experiments/task1/raw.json` 和 `summary.csv`，本次评测的 PNG/SVG 图表也保存在该目录。策略共用负载和轮询规则，换页额外使用有界主机页池；CPU tick 不是设备时间，容量拒绝不是 CUDA OOM。

GPU 对照运行 `python scripts/gpu_matrix.py`。详见[量化报告](../docs/02-量化评测报告.md)和[复现说明](../docs/04-复现与演示.md)。

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
