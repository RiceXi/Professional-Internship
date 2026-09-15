"""Logical pages, copy-on-write and bounded LRU paging; no CUDA imports."""
from __future__ import annotations

from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass, field

from .storage import ListStorage


class AllocationError(RuntimeError):
    """Configured capacity or pinned working set prevents an allocation."""


@dataclass
class Page:
    page_id: int
    frame: int | None
    refs: int = 1
    used: int = 0
    host: int | None = None
    touched: int = 0
    pins: int = 0


@dataclass
class AddressSpace:
    pages: list[int] = field(default_factory=list)
    length: int = 0


class PagedMemory:
    """Stable logical IDs; storage owns physical data and finishes each copy."""

    def __init__(self, num_frames: int, block_size: int, *, host_pages: int = 0, storage=None):
        self.storage = storage if storage is not None else ListStorage(num_frames, block_size, host_pages)
        if (self.storage.num_frames, self.storage.block_size, self.storage.host_pages) != (num_frames, block_size, host_pages):
            raise ValueError("storage geometry does not match memory geometry")
        self.num_frames, self.block_size, self.host_pages = num_frames, block_size, host_pages
        self.free_frames = deque(range(num_frames))
        self.free_host = deque(range(host_pages))
        self.pages: dict[int, Page] = {}
        self.sequences: dict[int, AddressSpace] = {}
        self._next_page = self._next_sequence = self._clock = 0
        self.cow_copies = self.swap_out_pages = self.swap_in_pages = self.host_cow_reads = 0

    def _touch(self, page: Page) -> None:
        self._clock += 1
        page.touched = self._clock

    def new_sequence(self) -> int:
        sid = self._next_sequence
        self._next_sequence += 1
        self.sequences[sid] = AddressSpace()
        return sid

    def fork(self, sid: int) -> int:
        parent = self.sequences[sid]
        child = self.new_sequence()
        self.sequences[child] = AddressSpace(list(parent.pages), parent.length)
        for pid in parent.pages:
            self.pages[pid].refs += 1
        return child

    def _victim(self, exclude: set[int] | None = None) -> Page:
        exclude = exclude or set()
        candidates = [p for p in self.pages.values()
                      if p.frame is not None and p.pins == 0 and p.page_id not in exclude]
        if not candidates:
            raise AllocationError("all candidate pages are pinned or in the working set")
        return min(candidates, key=lambda p: (p.touched, p.page_id))

    def swap_out(self, pid: int) -> None:
        page = self.pages[pid]
        if page.pins:
            raise AllocationError("cannot evict a pinned page")
        if page.frame is None:
            return
        if not self.free_host:
            raise AllocationError("host page pool exhausted")
        host, frame = self.free_host[0], page.frame
        self.storage.store(frame, host)
        self.free_host.popleft()
        self.free_frames.append(frame)
        page.frame, page.host = None, host
        self.swap_out_pages += 1

    def _acquire_frame(self) -> int:
        if not self.free_frames:
            self.swap_out(self._victim().page_id)
        return self.free_frames.popleft()

    def _check_growth(self, count: int) -> None:
        if count > len(self.free_frames) + len(self.free_host):
            raise AllocationError("combined device and host page capacity exhausted")
        if count and not self.free_frames:
            self._victim()  # Fail before changing mappings if all frames are pinned.

    def _new_page(self, source: Page | None = None) -> int:
        self._check_growth(1)
        frame = self._acquire_frame()
        try:
            if source is None:
                self.storage.clear(frame)
            elif source.frame is not None:
                self.storage.copy(source.frame, frame)
            else:
                self.storage.load(source.host, frame)
                self.host_cow_reads += 1
        except Exception:
            self.free_frames.appendleft(frame)
            raise
        pid = self._next_page
        self._next_page += 1
        page = Page(pid, frame, used=source.used if source else 0)
        self.pages[pid] = page
        self._touch(page)
        return pid

    def _make_private(self, seq: AddressSpace, index: int) -> None:
        old = self.pages[seq.pages[index]]
        if old.refs > 1:
            if old.pins:
                raise AllocationError("cannot replace a pinned shared mapping")
            pid = self._new_page(old)
            old.refs -= 1
            seq.pages[index] = pid
            self.cow_copies += 1

    def reserve(self, sid: int, length: int) -> None:
        """Grow without writing; capacity failures preserve sequence contents."""
        seq = self.sequences[sid]
        if length < seq.length:
            raise ValueError("reserve cannot shrink a sequence")
        count = (length + self.block_size - 1) // self.block_size - len(seq.pages)
        copy_tail = (length > seq.length and seq.length % self.block_size != 0
                     and self.pages[seq.pages[-1]].refs > 1)
        self._check_growth(count + int(copy_tail))
        previous = list(seq.pages)
        copies_before = self.cow_copies
        try:
            if copy_tail:
                self._make_private(seq, len(seq.pages) - 1)
            for _ in range(count):
                seq.pages.append(self._new_page())
        except Exception:
            # A backend allocation/copy can fail after capacity preflight.
            # Restore logical ownership; completed transfers may change residency.
            for pid in set(seq.pages) - set(previous):
                self._release_page(pid)
            for pid in set(previous) - set(seq.pages):
                self.pages[pid].refs += 1
            seq.pages = previous
            self.cow_copies = copies_before
            raise
        seq.length = length
        for i, pid in enumerate(seq.pages):
            self.pages[pid].used = min(self.block_size, length - i * self.block_size)

    def ensure_resident(self, page_ids) -> None:
        targets = set(page_ids)
        for pid in targets:
            self.pages[pid]  # Validate all IDs before any transfers.
        pinned_outside = sum(p.pins > 0 for pid, p in self.pages.items() if pid not in targets)
        if len(targets) + pinned_outside > self.num_frames:
            raise AllocationError("working set exceeds physical capacity; full attention requires resident KV")
        for pid in sorted(targets):
            page = self.pages[pid]
            if page.frame is None:
                host = page.host
                if self.free_frames:
                    frame = self.free_frames[0]
                    self.storage.load(host, frame)
                    self.free_frames.popleft()
                    self.storage.drop_host(host)
                    self.free_host.append(host)
                else:
                    victim = self._victim(targets)
                    frame = victim.frame
                    # Exchange reuses the incoming page's host slot, so a full
                    # host pool can still make progress. Storage uses one scratch page.
                    self.storage.exchange(frame, host)
                    victim.frame, victim.host = None, host
                    self.swap_out_pages += 1
                page.frame, page.host = frame, None
                self.swap_in_pages += 1
            self._touch(page)

    @contextmanager
    def pin_sequences(self, sids):
        ids = list(sids)
        targets = {pid for sid in ids for pid in self.sequences[sid].pages}
        self.ensure_resident(targets)
        for pid in targets:
            self.pages[pid].pins += 1
        try:
            yield {sid: [self.pages[pid].frame for pid in self.sequences[sid].pages] for sid in ids}
        finally:
            for pid in targets:
                self.pages[pid].pins -= 1

    def translate(self, sid: int, index: int) -> tuple[int, int]:
        seq = self.sequences[sid]
        if not 0 <= index < seq.length:
            raise IndexError("logical token address out of bounds")
        pid = seq.pages[index // self.block_size]
        self.ensure_resident([pid])
        return self.pages[pid].frame, index % self.block_size

    def write(self, sid: int, index: int, value) -> None:
        seq = self.sequences[sid]
        if not 0 <= index < seq.length:
            raise IndexError("logical token address out of bounds")
        self._make_private(seq, index // self.block_size)
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
        seq = self.sequences[sid]
        if any(self.pages[pid].pins for pid in seq.pages):
            raise AllocationError("cannot free pinned mappings")
        del self.sequences[sid]
        for pid in seq.pages:
            self._release_page(pid)

    def _release_page(self, pid: int) -> None:
        page = self.pages[pid]
        page.refs -= 1
        if page.refs == 0:
            if page.frame is not None:
                self.free_frames.append(page.frame)
            else:
                self.storage.drop_host(page.host)
                self.free_host.append(page.host)
            del self.pages[pid]

    def metrics(self) -> dict:
        allocated = len(self.pages) * self.block_size
        useful = sum(p.used for p in self.pages.values())
        resident = [p for p in self.pages.values() if p.frame is not None]
        logical = sum(s.length for s in self.sequences.values())
        return {
            "physical_capacity_tokens": self.num_frames * self.block_size,
            "resident_pages": len(resident), "host_pages_used": len(self.pages) - len(resident),
            "free_pages": len(self.free_frames), "free_host_pages": len(self.free_host),
            "allocated_slots": allocated, "useful_slots": useful, "logical_tokens": logical,
            "shared_slots_saved": logical - useful,
            "resident_useful_slots": sum(p.used for p in resident),
            "pool_occupancy": len(resident) / self.num_frames,
            "allocated_utilization": useful / allocated if allocated else 0.0,
            "internal_fragmentation": (allocated - useful) / allocated if allocated else 0.0,
            "cow_copies": self.cow_copies,
            "swap_out_pages": self.swap_out_pages, "swap_in_pages": self.swap_in_pages,
            "host_cow_reads": self.host_cow_reads,
        }

    def check_invariants(self) -> None:
        refs = Counter(pid for s in self.sequences.values() for pid in s.pages)
        assert set(refs) == set(self.pages)
        assert all(p.refs == refs[pid] and p.refs > 0 for pid, p in self.pages.items())
        assert all((p.frame is None) != (p.host is None) for p in self.pages.values())
        assert all(p.pins >= 0 and (not p.pins or p.frame is not None) for p in self.pages.values())
        frames = [p.frame for p in self.pages.values() if p.frame is not None]
        hosts = [p.host for p in self.pages.values() if p.host is not None]
        assert sorted(frames + list(self.free_frames)) == list(range(self.num_frames))
        assert sorted(hosts + list(self.free_host)) == list(range(self.host_pages))
        for seq in self.sequences.values():
            assert len(seq.pages) == (seq.length + self.block_size - 1) // self.block_size
            for i, pid in enumerate(seq.pages):
                assert self.pages[pid].used == min(self.block_size, seq.length - i * self.block_size)
