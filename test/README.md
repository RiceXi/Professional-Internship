# 测试说明

## CPU 课程测试

```bash
bash scripts/setup_cpu.sh
.venv-cpu/bin/python test/run_cpu.py --coverage
```

测试覆盖地址映射、跨页扩展、资源回收、分配失败原子性、连续内存碎片fork/COW 分支隔离、不同释放顺序以及确定性负载。覆盖率范围为 `src/memory`，要求 ≥70%。

## GPU 推理测试

```bash
python test/run_all.py
```

需要 CUDA 和本地 Qwen3-0.6B，覆盖引擎、采样、前缀缓存和混合调度。GPU 测试与 CPU 仿真测试分别报告；当前机器没有 CUDA，不能宣称 GPU 测试通过。
