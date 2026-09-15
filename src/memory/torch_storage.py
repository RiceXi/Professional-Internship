"""Tensor page storage. Same manager supports CPU tests and CUDA KV pools."""
from __future__ import annotations
import time
import torch


class TorchStorage:
    """Pool layout [2, layers, frames, block_size, kv_heads, head_dim].

    Transfers finish before returning so the manager can safely reuse page IDs.
    CUDA host pages are pinned. A full-pool exchange needs one extra host page
    temporarily; its size is exposed as exchange_scratch_bytes.
    """

    def __init__(self, pool: torch.Tensor, host_pages: int = 0):
        if pool.ndim != 6 or pool.shape[0] != 2 or any(n <= 0 for n in pool.shape):
            raise ValueError('expected nonempty [2,L,N,S,H,D] KV pool')
        if host_pages < 0:
            raise ValueError('host_pages must be nonnegative')
        self.pool = pool
        self.num_frames, self.block_size = pool.shape[2:4]
        self.host_pages = host_pages
        self.host = [None] * host_pages
        self.page_shape = (2, pool.shape[1], *pool.shape[3:])
        self.page_bytes = pool[:, :, 0].numel() * pool.element_size()
        self.exchange_scratch_bytes = self.page_bytes if host_pages else 0
        self.d2h_bytes = self.h2d_bytes = self.d2d_bytes = 0
        self.transfer_seconds = 0.0

    def _sync(self):
        if self.pool.is_cuda:
            torch.cuda.synchronize(self.pool.device)

    def _host_page(self):
        return torch.empty(self.page_shape, dtype=self.pool.dtype, device='cpu', pin_memory=self.pool.is_cuda)

    def clear(self, frame):
        self.pool[:, :, frame].zero_()
        self._sync()

    def copy(self, src, dst):
        t0 = time.perf_counter()
        self.pool[:, :, dst].copy_(self.pool[:, :, src])
        self._sync()
        self.d2d_bytes += self.page_bytes
        self.transfer_seconds += time.perf_counter() - t0

    def store(self, frame, host):
        t0 = time.perf_counter()
        page = self._host_page()
        page.copy_(self.pool[:, :, frame], non_blocking=False)
        self._sync()
        self.host[host] = page
        self.d2h_bytes += self.page_bytes
        self.transfer_seconds += time.perf_counter() - t0

    def load(self, host, frame):
        t0 = time.perf_counter()
        self.pool[:, :, frame].copy_(self.host[host], non_blocking=False)
        self._sync()
        self.h2d_bytes += self.page_bytes
        self.transfer_seconds += time.perf_counter() - t0

    def drop_host(self, host):
        self.host[host] = None

    def exchange(self, frame, host):
        t0 = time.perf_counter()
        scratch = self._host_page()
        scratch.copy_(self.pool[:, :, frame], non_blocking=False)
        self._sync()
        self.pool[:, :, frame].copy_(self.host[host], non_blocking=False)
        self._sync()
        self.host[host] = scratch
        self.d2h_bytes += self.page_bytes
        self.h2d_bytes += self.page_bytes
        self.transfer_seconds += time.perf_counter() - t0

    def write(self, frame, offset, value):
        value = torch.as_tensor(value, dtype=self.pool.dtype, device=self.pool.device)
        self.pool[:, :, frame, offset].copy_(value)
        self._sync()

    def read(self, frame, offset):
        return self.pool[:, :, frame, offset].detach().cpu().clone()

    def metrics(self):
        return {name: getattr(self, name) for name in
                ('page_bytes', 'exchange_scratch_bytes', 'd2h_bytes', 'h2d_bytes', 'd2d_bytes', 'transfer_seconds')}
