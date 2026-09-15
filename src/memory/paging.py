"""Logical pages and sequence page tables; no torch or CUDA imports."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

from .storage import ListStorage


class AllocationError(RuntimeError):
    """A request cannot fit within the configured memory capacity."""


@dataclass
class Page:
    page_id: int
    frame: int
    refs: int = 1
    used: int = 0


@dataclass
class AddressSpace:
    pages: list[int] = field(default_factory=list)
    length: int = 0


class PagedMemory:
    """Manage stable logical page IDs separately from physical frame IDs."""

    def __init__(self, num_frames: int, block_size: int, *, storage=None):
        self.storage = storage if storage is not None else ListStorage(num_frames, block_size)
        if (self.storage.num_frames, self.storage.block_size) != (num_frames, block_size):
            raise ValueError("storage geometry does not match memory geometry")
        self.num_frames = num_frames
        self.block_size = block_size
        self.free_frames = deque(range(num_frames))
        self.pages: dict[int, Page] = {}
        self.sequences: dict[int, AddressSpace] = {}
        self._next_page = 0
        self._next_sequence = 0

    def new_sequence(self) -> int:
        sid = self._next_sequence
        self._next_sequence += 1
        self.sequences[sid] = AddressSpace()
        return sid

    def reserve(self, sid: int, length: int) -> None:
        """Grow address space. Capacity failures leave its mappings unchanged."""
        seq = self.sequences[sid]
        if length < seq.length:
            raise ValueError("reserve cannot shrink a sequence")
        count = (length + self.block_size - 1) // self.block_size - len(seq.pages)
        if count > len(self.free_frames):
            raise AllocationError("physical page pool exhausted")
        for _ in range(count):
            frame = self.free_frames.popleft()
            self.storage.clear(frame)
            pid = self._next_page
            self._next_page += 1
            self.pages[pid] = Page(pid, frame)
            seq.pages.append(pid)
        seq.length = length
        for i, pid in enumerate(seq.pages):
            self.pages[pid].used = min(self.block_size, length - i * self.block_size)

    def translate(self, sid: int, index: int) -> tuple[int, int]:
        seq = self.sequences[sid]
        if not 0 <= index < seq.length:
            raise IndexError("logical token address out of bounds")
        pid = seq.pages[index // self.block_size]
        return self.pages[pid].frame, index % self.block_size

    def write(self, sid: int, index: int, value) -> None:
        frame, offset = self.translate(sid, index)
        self.storage.write(frame, offset, value)

    def append(self, sid: int, value) -> None:
        index = self.sequences[sid].length
        self.reserve(sid, index + 1)
        self.write(sid, index, value)

    def read(self, sid: int, index: int):
        frame, offset = self.translate(sid, index)
        return self.storage.read(frame, offset)

    def read_sequence(self, sid: int) -> list:
        return [self.read(sid, i) for i in range(self.sequences[sid].length)]

    def free(self, sid: int) -> None:
        seq = self.sequences.pop(sid)
        for pid in seq.pages:
            page = self.pages[pid]
            page.refs -= 1
            if page.refs == 0:
                self.free_frames.append(page.frame)
                del self.pages[pid]

    def metrics(self) -> dict:
        allocated = len(self.pages) * self.block_size
        useful = sum(p.used for p in self.pages.values())
        logical = sum(s.length for s in self.sequences.values())
        return {
            "physical_capacity_tokens": self.num_frames * self.block_size,
            "resident_pages": len(self.pages),
            "free_pages": len(self.free_frames),
            "allocated_slots": allocated,
            "useful_slots": useful,
            "logical_tokens": logical,
            "shared_slots_saved": logical - useful,
            "pool_occupancy": len(self.pages) / self.num_frames,
            "allocated_utilization": useful / allocated if allocated else 0.0,
            "internal_fragmentation": (allocated - useful) / allocated if allocated else 0.0,
        }

    def check_invariants(self) -> None:
        refs = Counter(pid for s in self.sequences.values() for pid in s.pages)
        assert set(refs) == set(self.pages)
        assert all(p.refs == refs[pid] and p.refs > 0 for pid, p in self.pages.items())
        frames = [p.frame for p in self.pages.values()]
        assert len(frames) == len(set(frames))
        assert not set(frames).intersection(self.free_frames)
        assert sorted(frames + list(self.free_frames)) == list(range(self.num_frames))
        for seq in self.sequences.values():
            assert len(seq.pages) == (seq.length + self.block_size - 1) // self.block_size
            for i, pid in enumerate(seq.pages):
                assert self.pages[pid].used == min(self.block_size, seq.length - i * self.block_size)
