# dzyy-vllm v3 — 算子级性能优化实践

从零实现的 Qwen3 推理引擎 v3。项目在 continuous batching 的基础上，围绕 Attention、KV Cache、CUDA Graph、`torch.compile`、算子融合和批量采样进行系统优化，适合作为大模型推理与 AI Infra 实习项目阅读和复现。

> 本仓库是 [`dzyy-team/dzyy-vllm`](https://github.com/dzyy-team/dzyy-vllm) 中 `vllm-v3` 的独立版本；代码、公共工具、测试和文档已整理为可单独克隆的结构。

## 核心能力

| 优化 | 实现位置 | 作用 |
|------|----------|------|
| FlashAttention paged Attention | `src/compute/layers/attention.py` | 直接使用 block table，避免显式 gather 和因果 mask |
| Decode 合批 | `src/compute/layers/attention.py` | 单次 kernel 处理整批 decode 请求 |
| Triton KV 写入 | `src/data/kv_ops.py` | fused scatter 替代 `index_copy_` |
| Decode CUDA Graph | `src/compute/cuda_graph.py` | 降低逐 token decode 的 CPU launch 开销 |
| Prefill `torch.compile` | `src/compute/model_runner.py` | 编译 prefill 首 chunk 的完整前向图 |
| 融合 QKV / Gate-Up GEMM | `src/compute/models/qwen3.py` | 减少 GEMM 次数和中间张量 |
| 批量采样 | `src/compute/sampler.py` | 将逐请求采样合并为批处理 |
| 快速权重加载 | `src/compute/models/loader.py` | 跳过随机初始化并预取 safetensors 分片 |

原项目实测中，Qwen3-0.6B decode 吞吐由 v2 的约 40 tok/s 提升到约 400 tok/s；Qwen3-4B 权重加载由约 29 秒缩短到约 3.9 秒。测试环境、负载和测量方法见[实测证据](docs/02-实测证据.md)，不同硬件上的结果会有差异。

## 仓库结构

```text
.
├── run.py                       # 单请求与批量推理入口
├── config.yaml                  # 模型、调度、KV Cache 与编译配置
├── model_hub.py                 # 模型下载和本地路径解析入口
├── src/
│   ├── control/                 # 控制面：Engine、Scheduler
│   ├── data/                    # 数据面：Batch、KV Cache、前缀缓存
│   └── compute/                 # 计算面：模型执行、Attention、CUDA Graph、Sampler
├── utils/                       # 配置、模型下载、权重加载等公共工具
├── test/                        # 功能正确性测试
├── bench/                       # 算子、机制和端到端性能基准
└── docs/                        # 设计与性能分析文档
```

## 环境要求

- Linux + NVIDIA CUDA GPU
- Python 3.12
- 与 CUDA、PyTorch 对齐的 FlashAttention 和 FlashInfer
- 建议从 Qwen3-0.6B 开始验证；CUDA Graph 和 compile 会额外占用显存

原开发环境使用 PyTorch 2.13.0、CUDA 13.0、FlashAttention 2.8.3 和 FlashInfer 0.6.15。GPU 扩展包必须与本机 Python、PyTorch 和 CUDA 版本匹配。

## 安装

以下命令以 `uv` 和 CUDA 13.0 为例：

```bash
git clone https://github.com/xixii421/Professional-Internship.git
cd Professional-Internship

uv venv .venv --python 3.12
source .venv/bin/activate

uv pip install torch==2.13.0 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu130

uv pip install flashinfer-python flashinfer-cubin \
  --index-url https://flashinfer.ai/whl \
  --extra-index-url https://pypi.org/simple
uv pip install flashinfer-jit-cache \
  --index-url https://flashinfer.ai/whl/cu130

uv pip install \
  https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.47/flash_attn-2.8.3+cu130torch2.13-cp312-cp312-linux_x86_64.whl

uv pip install -r requirements.txt
```

如果本机 CUDA 或 PyTorch 版本不同，请先从相应项目的官方安装说明中选择匹配的 wheel，再安装 `requirements.txt` 中的通用依赖。

## 模型准备

模型默认放在 `~/huggingface/`。可以用项目命令下载，也可以直接传入一个本地 Hugging Face 模型目录：

```bash
# 查看内置模型及按当前显存估算可运行的型号
python run.py --list

# 下载内置的 Qwen3-0.6B、1.7B 和 4B
python run.py --download

# 也可以自行下载
hf download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B
```

如需使用 Hugging Face Token 或镜像，可设置 `HF_TOKEN`、`HF_ENDPOINT` 环境变量。

## 快速开始

```bash
# eager 模式：适合首次跑通
python run.py --model Qwen3-0.6B --temperature 0

# Decode CUDA Graph
python run.py --model Qwen3-0.6B --cg

# Prefill torch.compile
python run.py --model Qwen3-0.6B --compile

# CUDA Graph + compile
python run.py --model Qwen3-0.6B --cg --compile

# 批量推理
python run.py --model Qwen3-0.6B --batch

# 直接使用本地模型路径
python run.py --model /path/to/Qwen3-0.6B
```

配置优先级为“命令行参数 > `config.yaml` > 代码默认值”。首次运行建议使用 eager；确认正确后再逐项打开 CUDA Graph 和 compile，便于定位依赖或显存问题。

## 测试与基准

功能测试需要可用的 CUDA 环境和本地 Qwen3-0.6B 模型：

```bash
python test/run_all.py
```

原生 v3 基准默认使用当前 Python 环境：

```bash
python bench/run_all.py
```

跨引擎对比还需要把 `vllm-v2`、`nano-vllm` 放在本仓库的同级目录，并为官方 vLLM 准备独立环境；具体命令和测量口径见 [bench/README.md](bench/README.md)。

## 文档导读

建议按编号阅读：00～02 解释为什么优化，03 以后说明如何实现。

| # | 文档 | 内容 |
|---|------|------|
| 00 | [优化总览](docs/00-优化总览.md) | v2 局限、三面框架与落地清单 |
| 01 | [v2 性能瓶颈](docs/01-v2性能瓶颈.md) | Attention、Decode、算子和传输瓶颈 |
| 02 | [实测证据](docs/02-实测证据.md) | 算子、机制、端到端三层对比 |
| 03 | [输入管线](docs/03-输入管线.md) | 变长请求打包与异步 H2D |
| 04 | [显存计算](docs/04-显存计算.md) | 权重、KV Cache 与编译显存预算 |
| 05 | [算子融合](docs/05-算子融合.md) | RMSNorm、RoPE、SiLU 和 GEMM 融合 |
| 06 | [Attention](docs/06-Attention.md) | FlashAttention paged attention 与 KV 写入 |
| 07 | [CUDA Graph](docs/07-CUDA-Graph.md) | Decode 图捕获与重放 |
| 08 | [Prefill 编译](docs/08-Prefill-Compile.md) | `torch.compile` 全图编译 |
| 09 | [Sampler](docs/09-Sampler.md) | 批量采样和同步开销 |
| 10 | [模型加载](docs/10-模型加载.md) | 快速权重加载 |

更多测试覆盖范围见 [test/README.md](test/README.md)，基准脚本说明见 [bench/README.md](bench/README.md)。
