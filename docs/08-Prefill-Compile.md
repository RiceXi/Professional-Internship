# Prefill 编译机制

本项目在 `src/compute/model_runner.py` 中用 `torch.compile` 包装无历史缓存的 Prefill 前向。Decode 由独立的 CUDA Graph 路径处理；两个开关可以分别配置。

## 1. 首 chunk 与续 chunk

Prefill 为本轮尚未缓存的 token 计算 Q/K/V。请求第一次执行且没有命中前缀时，`num_cached_tokens == 0`；因预算而分块执行后，下一段需要读取此前已计算的 KV。

例如一个 3000-token prompt 在每步 1024-token 预算下，可能按 1024、1024、952 分三次推进。第一段没有历史缓存，后两段仍要计算新 token，同时读取此前的 KV。命中共享前缀的请求在首次调度时也可能已经拥有历史缓存。

“续 chunk”是尚未计算的新段，不是已缓存的前缀本身。

## 2. 当前路径选择

`_run_prefill()` 只有在以下条件同时满足时调用编译函数：

- 配置启用 `torch_compile`，已经创建 `prefill_fn`。
- 本批每条序列均满足 `num_cached_tokens == 0`。

其他情况走 eager 前向。有历史缓存时，Attention 接收 `block_tables`，读取分页池中的历史 K/V。

```python
if self.prefill_fn is not None and all(s.num_cached_tokens == 0 for s in seqs):
    # 调用编译后的无历史缓存前向
    ...
else:
    # eager 前向；存在历史缓存时使用分页 Attention
    ...
```

## 3. 自定义算子如何参与编译

FlashAttention 和 FlashInfer 由底层内核执行。项目把 Attention、RoPE、SiLU、RMSNorm 等封装成 `torch.library.custom_op`，并提供 fake 实现，向编译器说明输出形状、数据类型和输入修改情况。

编译器可将这些调用作为图节点处理；底层内核内部不会因此自动获得跨算子融合。`fused_add_rmsnorm` 会原地修改输入与残差，需要在 `mutates_args` 中声明。

当前 `torch.compile` 调用传入 `mode` 和 `dynamic`，没有设置 `fullgraph=True`。是否出现图切分、生成多少图和内核，应由实际编译日志或 profiler 验证，不能仅凭开启配置就宣称“全图编译成功”。

## 4. 动态形状与启动代价

`compile_dynamic=True` 尝试使用动态形状，降低不同输入长度触发重编译的概率；它不保证所有 batch 大小、标量参数或执行条件都共用一份编译产物。

初始化阶段会用一个短输入预热，再清空临时写入的 KV。这个预热不覆盖所有后续输入，因此实验应分别统计初始化、首次遇到新形状、稳定重复运行三个阶段。

## 5. 验证方法

- 使用 `test/test_engine.py` 检查选定样本的 eager、CUDA Graph 和编译输出。
- 使用 `bench/bench_prefill.py` 比较不同 prompt 长度，保持其他执行开关与采样参数一致。
- 单独覆盖无缓存首段、续段、共享前缀和混合批次，确认路径选择正确。
- 记录重编译次数、图切分、首 token 延迟与峰值显存。

浮点运算顺序变化可能影响接近并列的 logits。若生成结果不同，应定位误差、检查容差和任务效果；不能仅以“低精度波动”为理由忽略失败。当前测试对其样本要求生成 token 一致，这不是所有输入上的等价性证明。
