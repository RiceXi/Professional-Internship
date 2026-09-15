# 功能正确性测试

功能正确性测试，与性能 benchmark 分离。每个脚本独立可跑，任一断言失败返回非零退出码。

需要 CUDA 与本地 `~/huggingface/Qwen3-0.6B`；当前不是无 GPU 仿真测试。批量运行器优先选择名称包含 4090 的设备，否则选择设备 0。

在仓库根目录下：

```bash
# 全部
python test/run_all.py

# 或逐个
python test/test_sampler.py
python test/test_engine.py
python test/test_prefix.py
python test/test_scheduler.py
```

| 脚本 | 覆盖模块 | 验证内容 |
|------|----------|----------|
| `test_sampler.py` | compute/sampler | greedy / 纯温度 / top_k / top_p / min_p / 混合 六分支正确性、固定 seed 可复现 |
| `test_engine.py` | control/engine、compute/model_runner、compute/cuda_graph | eager 可复现、decode CG 与 eager 等价、compile+CG 与 eager 等价、max_tokens 终止、prompt 截断、max_tokens 收敛、EOS smoke |
| `test_prefix.py` | data/prefix（hash/radix）、data/manager、data/block_manager | 完整/部分前缀命中后输出与无缓存一致（不能据此证明通用 COW） |
| `test_scheduler.py` | control/scheduler | mix 开/关时 dec@P 行为、长请求 chunk prefill 完成 |

公共件在 `common.py`：引擎工厂、等长 dummy、greedy 运行、warmup 与销毁。不含计时逻辑，计时归 `bench/`。

## 尚未覆盖的课程要求

目前未提供仿真器覆盖率采集，不能声称达到任务书要求的 ≥ 70%。分配回收不变量、共享块写入隔离、换入换出、资源耗尽恢复等需要独立用例；若选择调度任务，还需要动态到达、策略、抢占恢复与防饥饿测试。

引擎测试主要比较本实现不同执行模式，尚未提供与独立参考模型的 logits/生成结果对照。EOS 用例为 smoke 测试，不强制实际命中 EOS。
