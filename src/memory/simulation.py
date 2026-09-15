"""Deterministic arrivals, one-token iterations, and bounded request lifetimes."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import random

from .contiguous import ContiguousMemory
from .paging import AllocationError, PagedMemory


@dataclass(frozen=True)
class Request:
    rid: int
    arrival: int
    tokens: int
    maximum: int


def workload(seed: int = 42, count: int = 32) -> list[Request]:
    if count <= 0:
        raise ValueError('count must be positive')
    rng = random.Random(seed)
    return [Request(i, i // 4, rng.randint(3, 20), 32) for i in range(count)]


def simulate(requests: list[Request], mode: str, *, frames: int = 32, block_size: int = 4, host_pages: int = 32) -> dict:
    if mode not in ('contiguous', 'paged', 'swap'):
        raise ValueError('unknown memory mode')
    if frames <= 0 or block_size <= 0:
        raise ValueError('capacity must be positive')
    if len({r.rid for r in requests}) != len(requests):
        raise ValueError('duplicate request IDs')
    for r in requests:
        if r.arrival < 0 or not 0 < r.tokens <= r.maximum:
            raise ValueError('invalid request')
    memory = ContiguousMemory(frames * block_size) if mode == 'contiguous' else PagedMemory(frames, block_size, host_pages=host_pages if mode == 'swap' else 0)
    pending = sorted(requests, key=lambda r: (r.arrival, r.rid))
    active = {}
    records = []
    trace = []
    t = 0
    while pending or active:
        if not active and pending:
            t = max(t, pending[0].arrival)
        while pending and pending[0].arrival <= t:
            r = pending.pop(0)
            try:
                sid = memory.new_sequence(r.maximum) if mode == 'contiguous' else memory.new_sequence()
                active[r.rid] = [r, sid, 0]
            except AllocationError:
                records.append({'rid': r.rid, 'status': 'capacity_rejected', 'at': t})
        done = []
        for rid, (r, sid, used) in sorted(active.items()):
            try:
                memory.reserve(sid, used + 1)
                if mode != 'contiguous':
                    with memory.pin_sequences([sid]):
                        pass  # A full attention working set must fit physical memory.
                active[rid][2] += 1
                if used + 1 == r.tokens:
                    records.append({'rid': rid, 'status': 'completed', 'at': t})
                    done.append(rid)
            except AllocationError:
                records.append({'rid': rid, 'status': 'capacity_rejected', 'at': t})
                done.append(rid)
        trace.append({'tick': t, 'active': len(active), **memory.metrics()})
        for rid in done:
            memory.free(active.pop(rid)[1])
        memory.check_invariants()
        t += 1
    return {
        'kind': 'cpu_simulation', 'mode': mode,
        'workload': [asdict(r) for r in requests],
        'frames': frames, 'block_size': block_size,
        'host_pages': host_pages if mode == 'swap' else 0,
        'completed': sum(r['status'] == 'completed' for r in records),
        'capacity_rejected': sum(r['status'] == 'capacity_rejected' for r in records),
        'records': sorted(records, key=lambda r: r['rid']), 'trace': trace,
        'final': memory.metrics(),
    }
