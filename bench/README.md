# 性能基准与对照实验

性能测试，与功能正确性 test 分离。当前结果写入 `bench/out/*.txt`。该目录被 Git 忽略，批量运行前会清空，需要留作课程证据的输出应另行归档。

需要 CUDA 与本地 `~/huggingface/Qwen3-0.6B`。运行器优先选择名称包含 4090 的设备，否则选择设备 0。

在仓库根目录下：

```bash
# 本项目默认基准：清空 out/ 后顺序跑
python bench/run_all.py

# 追加 nano-vllm / 官方 vLLM 对比
# nano-vllm 需要放在本仓库同级目录；官方 vLLM 使用独立环境
VLLM_PYTHON=/path/to/vllm-env/bin/python python bench/run_all.py --with-external

# 或逐个
python bench/bench_prefill.py
python bench/bench_e2e.py
python bench/bench_compare.py --engine v3
python bench/bench_compare.py --engine nano
/path/to/vllm-env/bin/python bench/bench_compare.py --engine vllm
python bench/bench_ttft_decode.py --engine v3
python bench/bench_ttft_decode.py --engine nano
/path/to/vllm-env/bin/python bench/bench_ttft_decode.py --engine vllm
python bench/bench_concurrent.py --engine v3
python bench/bench_concurrent.py --engine nano
/path/to/vllm-env/bin/python bench/bench_concurrent.py --engine vllm
```

单项脚本的 `--engine v3` 是目前代码保留的本项目引擎标识。它不要求安装其他版本；运行默认套件无需提供该参数。外部引擎是可选对照。

| 脚本 | 回答什么 | 覆盖模块 | 输出 |
|------|----------|----------|------|
| bench_prefill.py | 短/长 prompt 稳态 TTFT：eager vs compile | prefill / torch.compile | out/bench_prefill.txt |
| bench_e2e.py | eager / decode-CG / CG+compile 的连续批吞吐（隔离 CG 与 compile 边际收益） | decode CUDA Graph + compile | out/bench_e2e.txt |
| bench_compare.py | 同一 workload 对比本项目、nano-vllm、官方 vLLM 吞吐 | 端到端跨引擎 | out/bench_compare.txt |
| bench_ttft_decode.py | 规范 TTFT / TPOT；本项目 radix vs nano/官方 hash 前缀命中 | TTFT/TPOT/前缀 | out/bench_ttft_decode.txt |
| bench_concurrent.py | 高并发 32-256（gpu_mem=0.90）吞吐对比 | 高并发连续批 | out/bench_concurrent.txt |

共享逻辑在 common.py：引擎工厂、等长 dummy、3 轮 greedy warmup（compile -> 录图 -> 重放），计时不含 tokenizer/print。输出统一为 txt：write_result 覆盖式写单个报告，append_lines 供跨引擎脚本各自追加一行、header 只写一次。

口径：

- TTFT = 第一个 completion token；greedy；warmup 之后取中位数
- Decode / 连续批吞吐见 bench_e2e.py、bench_compare.py
- bench_prefill 两边都开 decode CG，只隔离 torch_compile
- bench_ttft_decode 因 nano 禁止 greedy，三边统一 temperature=0.6、ignore_eos；TTFT / TPOT 对齐 vLLM bench serve（并发 1；prefix 场景 prefix=1024 / suffix=128）
- 三个引擎不能同进程混 import，跨引擎脚本按 --engine 分三次跑

## 辅助脚本与课程评测边界

默认套件只运行上表列出的五项脚本。其余 `bench_ops.py`、`bench_kvcache_write.py`、`bench_decode_attention.py` 和 RMSNorm 脚本用于辅助微基准；部分辅助对照仍引用外部实现路径，应检查各脚本的依赖后再运行。

当前套件主要衡量 GPU 执行性能。分页任务还缺连续分配基线、碎片/利用率/OOM 统计；调度任务还缺静态批处理基线、到达流、多策略与 P50/P99、公平性统计。报告需要保存环境、配置、负载、原始输出和绘图过程，不能仅引用文档中的历史数字。
