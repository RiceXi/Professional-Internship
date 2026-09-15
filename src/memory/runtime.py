"""Iteration driver for CPU or real model callbacks, with identical page rules."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from .paging import AllocationError, PagedMemory


@dataclass
class InferenceRequest:
    rid: int
    memory_id: int
    tokens: list[int]
    prompt_length: int
    max_new_tokens: int
    cached: int = 0
    status: str = 'running'
    error: str | None = None

    @property
    def completion(self):
        return self.tokens[self.prompt_length:]


class MemoryRuntime:
    """Round-robin one-request steps isolate memory policy from scheduling.

    execute(request, physical_table, start, count, need_sample) must finish its
    writes before returning a token (or None for a non-sampling chunk).
    """

    def __init__(self, memory: PagedMemory, execute, *, chunk_size: int = 256,
                 context_len: int = 4096, eos_token_id: int | None = None):
        if chunk_size <= 0 or context_len <= 0:
            raise ValueError('chunk_size and context_len must be positive')
        self.memory, self.execute = memory, execute
        self.chunk_size, self.context_len, self.eos_token_id = chunk_size, context_len, eos_token_id
        self.requests: dict[int, InferenceRequest] = {}
        self.queue = deque()
        self.history = []
        self._next_id = 0

    def add(self, prompt: list[int], max_new_tokens: int = 16) -> int:
        if not prompt or max_new_tokens < 0:
            raise ValueError('nonempty prompt and nonnegative output length required')
        if len(prompt) + max_new_tokens > self.context_len:
            raise ValueError('request exceeds configured context length')
        rid = self._next_id
        self._next_id += 1
        self.requests[rid] = InferenceRequest(rid, self.memory.new_sequence(), list(prompt), len(prompt), max_new_tokens)
        self.queue.append(rid)
        return rid

    def fork(self, rid: int) -> int:
        parent = self.requests[rid]
        if parent.status != 'running':
            raise ValueError('only a live request can be forked')
        child = self._next_id
        self._next_id += 1
        self.requests[child] = InferenceRequest(child, self.memory.fork(parent.memory_id),
            list(parent.tokens), parent.prompt_length, parent.max_new_tokens, parent.cached)
        self.queue.append(child)
        return child

    def cancel(self, rid: int) -> None:
        req = self.requests[rid]
        if req.status != 'running':
            raise ValueError('request is not running')
        self.memory.free(req.memory_id)
        self.queue.remove(rid)
        req.status = 'cancelled'

    def step(self) -> int | None:
        if not self.queue:
            return None
        rid = self.queue.popleft()
        req = self.requests[rid]
        count = min(self.chunk_size, len(req.tokens) - req.cached)
        end = req.cached + count
        need_sample = end >= req.prompt_length and req.max_new_tokens > 0
        try:
            if (end + self.memory.block_size - 1) // self.memory.block_size > self.memory.num_frames:
                raise AllocationError('request attention working set exceeds physical KV capacity')
            self.memory.reserve(req.memory_id, end)
            with self.memory.pin_sequences([req.memory_id]) as tables:
                token = self.execute(req, tables[req.memory_id], req.cached, count, need_sample)
            if need_sample and token is None:
                raise RuntimeError('executor did not return a token')
            req.cached = end
            if need_sample:
                req.tokens.append(int(token))
            if len(req.completion) >= req.max_new_tokens or (need_sample and token == self.eos_token_id):
                # A zero-output request must still finish all its prefill chunks.
                if req.cached >= req.prompt_length:
                    req.status = 'completed'
        except AllocationError as error:
            req.status, req.error = 'capacity_rejected', str(error)
        except Exception as error:
            req.status, req.error = 'execution_failed', str(error)
            self.memory.free(req.memory_id)
            raise
        self.history.append({'step': len(self.history), 'rid': rid, **self.memory.metrics()})
        if req.status == 'running':
            self.queue.append(rid)
        else:
            self.memory.free(req.memory_id)
        self.memory.check_invariants()
        return rid

    def run(self, max_steps: int = 100000) -> dict[int, list[int]]:
        for _ in range(max_steps):
            if not self.queue:
                return {rid: r.completion for rid, r in self.requests.items() if r.status == 'completed'}
            self.step()
        if self.queue:
            raise RuntimeError('step budget exhausted')
        return {rid: r.completion for rid, r in self.requests.items() if r.status == 'completed'}
