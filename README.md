# Professional Internship · KV Cache 分页内存管理

本项目用于专业实习课程，选择任务一「KV Cache 分页内存管理子系统」验收。以 Qwen3 推理引擎为实验载体，研究逻辑页表、按需分配、前缀共享、写时复制与页面置换，分析分页管理对显存利用率、碎片、并发容量和 OOM 的影响。

课程目标是形成可复现的系统实验：解释机制、验证正确性，并用基线对照和量化指标分析收益及代价。当前仓库包含 GPU 推理实现、功能测试和性能基准，课程仿真与验收材料仍需完善。

## 课程任务与当前范围

验收依据为仓库中的[大模型推理优化任务书](./面向大模型推理优化的操作系统级任务设计_合并版.docx)第二部分「任务一」及第六部分「通用考核与交付物」。任务书按 12 周、96 学时设计。

| 验收内容 | 当前基础 | 尚需完成的关键内容 |
|---|---|---|
| 块分配、页表与按需扩页 | 已有块池、引用计数与逻辑位置映射 | 无 GPU 仿真入口、边界与资源回收单测 |
| 共享与写时复制 | 已有整块前缀复用 | 序列 fork、未满共享块的写入复制与隔离验证 |
| 页面置换 | 已有缓存淘汰与压力下重算 | LRU 换出、CPU 数据保存、换入恢复 |
| 量化对照 | 已有 GPU 性能脚本 | 连续分配基线、利用率/碎片/并发容量/OOM 实验及图表 |
| 可复现交付 | 已有推理入口和模块说明 | 环境启动、覆盖率、设计与评测报告、答辩材料 |

现有调度器与 GPU 算子为分页管理提供运行和验证支撑。任务二至四的独立机制不列入本次验收。表中待完成项尚未实现，分阶段补齐建议见[任务一验收与补齐方案](docs/11-任务一验收与补齐方案.md)。

## 系统结构与已有能力

| 层次 | 模块 | 已有机制 |
|---|---|---|
| 控制面 | `src/control/engine.py`、`scheduler.py` | 请求生命周期、Continuous Batching、Chunked Prefill、Prefill/Decode 混合调度 |
| 数据面 | `src/data/manager.py`、`block_manager.py`、`sequence.py` | 分页 KV 池、逻辑位置到物理槽位映射、引用计数、按块扩展与回收 |
| 数据面 | `src/data/prefix/` | `none` / `hash` / `radix` 前缀缓存后端 |
| 数据面 | `src/data/batch.py`、`kv_ops.py` | 变长请求打包、锁页内存拷贝、Triton KV 写入 |
| 计算面 | `src/compute/layers/`、`models/` | FlashAttention 分页读写路径、FlashInfer 算子、QKV 与 Gate-Up 投影合并 |
| 计算面 | `src/compute/cuda_graph.py`、`model_runner.py` | Decode CUDA Graph、无历史缓存的 Prefill 首 chunk 编译 |
| 计算面 | `src/compute/sampler.py` | 批量采样、贪心快路径、top-k / top-p / min-p |
| 工程支持 | `utils/model_loader.py` | 跳过随机初始化、按目标精度构建、safetensors 分片预取 |

一次执行由 `Engine.step()` 驱动：调度器选择请求，数据层准备 token、页表与槽位，计算层执行前向和采样，随后更新请求状态并回收已完成请求的 KV 块。

### 实现边界

- 前缀复用以完整块为单位；尚未提供通用序列 fork 和未满共享块的写时复制。
- KV 压力抢占会释放块并在后续重算；缓存淘汰尚未实现数据向 CPU 的换出与恢复。
- 混合调度让 Decode 和 Prefill 在同一步推进；计算层依次执行两条路径。
- 权重分片预取用于模型加载阶段；当前模型仍整体驻留 GPU。
- 当前推理与测试入口依赖 CUDA；尚无开箱即用的纯 CPU 课程仿真入口。

## 仓库结构

```text
.
├── 面向大模型推理优化的操作系统级任务设计_合并版.docx
├── run.py                       # 单请求流式生成与批量推理
├── config.yaml                  # 模型、调度、KV Cache 与执行配置
├── model_hub.py                 # 模型下载和本地路径解析入口
├── src/
│   ├── control/                 # Engine、Scheduler
│   ├── data/                    # Sequence、Batch、KV 管理与前缀索引
│   └── compute/                 # 模型执行、算子、CUDA Graph、Sampler
├── utils/                       # 配置、模型下载与权重加载
├── test/                        # GPU 功能正确性测试
├── bench/                       # 算子、机制和端到端性能基准
└── docs/                        # 系统设计与推理机制讲解
```

## GPU 实验环境

当前推理实现面向 Linux + NVIDIA CUDA GPU，使用 Python 3.12 环境。PyTorch、FlashAttention、FlashInfer 必须与所选 GPU、CUDA 和 Python 环境匹配。实际支持范围以各依赖的官方说明为准；本仓库尚未提供经过独立复现验证的环境锁定文件。

### 安装步骤

```bash
git clone https://github.com/xixii421/Professional-Internship.git
cd Professional-Internship

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

在该虚拟环境内依次完成以下安装：

1. 按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)选择 Linux、Pip 和对应 CUDA 构建。
2. 按 [FlashAttention 官方说明](https://github.com/Dao-AILab/flash-attention#installation-and-features)安装提供 `flash_attn` 接口的兼容版本，并核对 GPU 架构要求。
3. 按 [FlashInfer 官方说明](https://docs.flashinfer.ai/installation.html)安装兼容版本。
4. 安装本项目通用依赖：

```bash
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch, flash_attn, flashinfer; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()"
```

上述检查仅确认依赖可导入和 CUDA 可用；模型与算子兼容性还需通过后面的功能测试验证。

## 模型准备

模型默认放在 `~/huggingface/`。安装好 GPU 依赖后，可以使用项目命令，也可以传入本地 Hugging Face 模型目录：

```bash
# 查看内置模型及显存估算
python run.py --list

# 下载内置的 Qwen3-0.6B、1.7B 和 4B
python run.py --download

# 仅准备 Qwen3-0.6B
hf download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B
```

如需使用 Hugging Face Token 或镜像，可设置 `HF_TOKEN`、`HF_ENDPOINT`。

## 快速开始

```bash
# 首次验证：贪心生成、eager 执行
python run.py --model Qwen3-0.6B --temperature 0 --eager --no-compile

# Decode CUDA Graph
python run.py --model Qwen3-0.6B --cg

# Prefill 编译
python run.py --model Qwen3-0.6B --compile

# CUDA Graph + Prefill 编译
python run.py --model Qwen3-0.6B --cg --compile

# 批量推理
python run.py --model Qwen3-0.6B --batch

# 本地模型路径
python run.py --model /path/to/Qwen3-0.6B
```

配置优先级为“命令行参数 > `config.yaml` > 代码默认值”。`enforce_eager` 控制 Decode CUDA Graph，`torch_compile` 单独控制 Prefill 编译。编译与图捕获会增加启动时间和显存占用，实验应分别记录冷启动与预热后的性能。

## 测试与实验

当前功能测试需要 CUDA 和本地 `~/huggingface/Qwen3-0.6B`：

```bash
python test/run_all.py
```

运行本项目的性能基准：

```bash
python bench/run_all.py
```

基准脚本覆盖 Prefill、Decode、前缀缓存与并发吞吐。默认基准不要求其他引擎仓库；外部引擎对比是可选实验，见 [bench/README.md](bench/README.md)。完整覆盖范围见 [test/README.md](test/README.md)。

当前两个批量运行器会优先选择名称包含 4090 的设备，否则选择设备 0。基准运行器会清空 `bench/out/`，需要保留的结果应在重新运行前另行归档。

### 课程实验应回答的问题

- 相同请求负载下，分页分配与连续预分配的显存利用率、碎片率和 OOM 比例有何差异？
- 前缀共享节省了多少物理块，引用释放和后续写入是否正确？
- 显存不足时，淘汰、重算和换出分别付出什么代价？
- 固定模型、上下文和物理容量后，分页管理能承载多少并发请求，长上下文何时触发 OOM？
- 在固定 GPU 执行配置下，内存管理机制的空间收益会带来多少换页延迟与吞吐代价？

上述问题中仍有待实现的基线和机制。课程结果应来自固定环境、固定负载、可追溯原始数据的实际运行。已有[参考测量记录](docs/02-实测证据.md)尚缺独立复现证据，不作为本项目课程验收的性能结论。

## 文档导读

文档按“系统目标 → 瓶颈原理 → 验证方法 → 模块实现”组织。

| 文档 | 内容 |
|---|---|
| [任务一验收与补齐方案](docs/11-任务一验收与补齐方案.md) | 已确定的验收范围、待商议的实施阶段与验收标准 |
| [系统与优化总览](docs/00-优化总览.md) | 课程定位、模块职责、OS 概念映射与实现边界 |
| [推理性能瓶颈](docs/01-推理性能瓶颈.md) | 朴素执行路径的访存、算子启动与同步开销 |
| [评测方法与参考记录](docs/02-实测证据.md) | 测量口径、待复现数据与课程评测缺口 |
| [输入管线](docs/03-输入管线.md) | 变长请求打包、页表与 H2D/D2H |
| [显存计算](docs/04-显存计算.md) | 权重、分页 KV 与运行时显存预算 |
| [算子融合](docs/05-算子融合.md) | RMSNorm、RoPE、SiLU 和 GEMM |
| [Attention](docs/06-Attention.md) | 分块计算、在线 softmax 与分页 KV 访问 |
| [CUDA Graph](docs/07-CUDA-Graph.md) | Decode 图捕获与重放 |
| [Prefill 编译](docs/08-Prefill-Compile.md) | 编译路径、适用条件与验证 |
| [Sampler](docs/09-Sampler.md) | 批量采样与候选集过滤 |
| [模型加载](docs/10-模型加载.md) | 初始化、分片预取与权重映射 |

## 课程交付进度

- 已有：推理代码、模块说明、GPU 功能测试、性能基准脚本。
- 待完善：无 GPU 仿真与单测覆盖率报告（任务书要求 ≥ 70%）、环境一键启动脚本、COW、换页与连续分配基线。
- 待形成：系统设计文档中的模块图与时序图、可复现的量化评测报告和图表。
- 需按实际过程记录：小组分工、每周例会、代码互审、答辩 PPT 与 5 分钟演示录屏。
