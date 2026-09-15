# 批量采样机制

`src/compute/sampler.py` 接收 `[batch_size, vocab_size]` 的 logits，一次返回每条请求的下一 token。`Batch._fill_sampling()` 在 CPU 上整理温度、top-k、top-p、min-p 和批次分支开关，避免为每个请求单独调用采样器。

## 1. 分支选择

| 场景 | 当前路径 |
|---|---|
| 全部贪心 | 直接 `argmax` |
| 仅温度采样 | 概率与指数随机变量构造的 Gumbel-Max 等价采样 |
| 含 top-k | 先取整批最大 k 的候选，再屏蔽各行多余列 |
| 无 top-k、含 top-p | 按整词表降序排列，再按累计概率过滤 |
| 仅 min-p | 根据该行最大概率的比例阈值过滤 |
| 贪心与随机混合 | 用安全温度执行批处理，最后将贪心行覆盖为 `argmax` |

贪心阈值在当前实现中为 `temperature <= 1e-5`。项目默认配置的温度为 0.9；需要确定性实验时显式设置为 0。

## 2. 候选空间缩减

```python
vals, idx = torch.topk(scaled, max_top_k, dim=-1)
col = torch.arange(max_top_k, device=logits.device)
vals = vals.masked_fill(col.unsqueeze(0) >= top_ks.unsqueeze(1), float("-inf"))
```

`vals` 保存候选分数，`idx` 保存原 token 编号。后续在候选空间执行 softmax、top-p/min-p 过滤与 `multinomial`，最后通过 `idx.gather()` 返回 token。

收益取决于候选列数相对词表的缩减程度。如果批内有请求禁用 top-k，其 k 会归一为词表大小，整批 `max_top_k` 也可能达到词表大小，不能假定所有混合批次都只处理少量候选。

## 3. top-p 与 min-p

`_sample_ordered()` 在降序概率上计算累计和。top-p 保留达到阈值所需的前缀候选；min-p 丢弃低于最大候选概率乘比例阈值的项。实现保留第一候选，避免过滤后支撑集为空。

启用 top-k 时，这些操作在 top-k 后的候选集合内执行。对照实现必须使用相同过滤顺序与归一化口径。

## 4. 纯温度采样

```python
scaled = logits.float() / temperatures.unsqueeze(1)
probs = torch.softmax(scaled, dim=-1)
noise = torch.empty_like(probs).exponential_().clamp_min_(1e-10)
tokens = probs.div_(noise).argmax(dim=-1)
```

为各候选生成独立的指数随机变量 E_i，选择 `p_i / E_i` 最大者，等价于在对数概率上加 Gumbel 噪声后取最大值。该函数当前采用 eager 执行。

需要区分“本实现用哪个算法”和“算法能否实现过滤”：Gumbel 方法也可用于已过滤的概率分布，本项目在过滤分支选择了 `multinomial`。

## 5. 结果回传与正确性边界

Sampler 返回 GPU token 张量。`ModelRunner` 将结果批量拷入锁页 CPU 缓冲，记录完成事件；`Engine.step()` 随即读取结果并推进调度。这样避免逐请求 `.item()`，但当前控制流程仍需每步等待采样结果。

当前实现会以 float32 计算温度缩放与概率，并为贪心行避免除零；代码没有通用 `nan_to_num` 清理路径，不能宣称任意 NaN/Inf 输入都能正常处理。纯温度分支使用全局随机数状态，传入的 `generator` 只在 `multinomial` 分支使用。

## 6. 如何验证

现有 `test/test_sampler.py` 覆盖贪心、温度、top-k、top-p、min-p 和混合参数。实验需要固定随机种子，并分别检查合法候选、过滤顺序、混合贪心行和重复运行行为。

采样属于执行层机制；课程调度实验仍应独立度量请求等待、尾延迟和公平性。采样加速不能替代这些指标。
