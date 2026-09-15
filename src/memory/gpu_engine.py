"""Course memory runtime connected to the repository's Qwen3 executor."""
from __future__ import annotations

from .paging import PagedMemory
from .runtime import MemoryRuntime


class CourseEngine:
    def __init__(self, config, *, host_pages: int = 16, chunk_size: int = 256, ignore_eos: bool = True):
        if not config.enforce_eager or config.torch_compile:
            raise ValueError('course comparisons currently require eager mode without compile')
        from src.control.engine import Engine
        from .torch_storage import TorchStorage
        self.base = Engine(config)
        kv = self.base.kv
        self.storage = TorchStorage(kv.kv_cache, host_pages)
        self.memory = PagedMemory(kv.kv_cache.shape[2], kv.block_size, host_pages=host_pages, storage=self.storage)
        self.runtime = MemoryRuntime(self.memory, self._execute,
            chunk_size=min(chunk_size, config.max_num_batched_tokens),
            context_len=config.context_len,
            eos_token_id=None if ignore_eos else self.base.eos_token_id)

    def _execute(self, request, table, start, count, need_sample):
        from src.data import ScheduledBatch, Sequence
        seq = Sequence(token_ids=request.tokens, block_size=self.memory.block_size,
            seq_id=request.rid, block_table=table, num_cached_tokens=start,
            num_scheduled_tokens=count, num_prompt_tokens=request.prompt_length,
            need_sample=need_sample, last_token=request.tokens[-1], temperature=0.0)
        is_prefill = start < request.prompt_length
        batch = ScheduledBatch(prefills=[seq] if is_prefill else [], decodes=[] if is_prefill else [seq])
        n_p, n_d = self.base.executor.run(batch)
        p_ids, d_ids = self.base.executor.read_samples(n_p, n_d)
        return (p_ids if is_prefill else d_ids)[0] if need_sample else None

    def add(self, prompt, max_new_tokens=16):
        ids = self.base.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        return self.runtime.add(ids, max_new_tokens)

    def destroy(self):
        for rid in list(self.runtime.queue):
            self.runtime.cancel(rid)
        # Drop all pool aliases before the executor destroys its CUDA allocation.
        self.runtime.execute = None
        self.memory.storage = None
        self.storage = None
        self.memory = None
        self.runtime = None
        self.base.destroy()
        self.base = None
