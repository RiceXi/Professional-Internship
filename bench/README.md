# 量化对照实验

## 课程内存实验

```bash
python3 bench/bench_memory.py
.venv-cpu/bin/python -m pip install -r requirements-report.txt
.venv-cpu/bin/python scripts/plot_memory.py
python3 scripts/report_memory.py
```

共 129 组确定性 CPU 仿真，对照连续预分配、按需分页、分页与主机换页；另有共享分支 COW 和外部碎片用例。记录页大小、物理容量、负载、每步资源状态、完成与容量拒绝，以及源码哈希。

输出为 `experiments/task1/raw.json`、`summary.csv` 和 PNG/SVG 图表。策略共用负载和轮询规则，换页额外使用有界主机页池；CPU tick 不是设备时间，容量拒绝不是 CUDA OOM。

真实 CUDA 对照运行 `python scripts/gpu_matrix.py`。详见[量化报告](../docs/02-量化评测报告.md)和[复现说明](../docs/04-复现与演示.md)。

## GPU 执行性能辅助基准

现有性能套件保留作推理支撑模块的独立测量，需要 CUDA 和本地模型。默认使用设备 0，遵守 `CUDA_VISIBLE_DEVICES`。

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

这些基准改变的是计算执行配置，不能直接当成课程分页/COW/换页的收益证据。本次量化报告使用第一节的内存对照脚本。
