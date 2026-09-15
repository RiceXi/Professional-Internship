"""First-fit contiguous reservation baseline with explicit free extents."""
from __future__ import annotations
from .paging import AllocationError


class ContiguousMemory:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.holes = [(0, capacity)]
        self.allocations: dict[int, tuple[int, int, int]] = {}
        self._next_sequence = 0

    def new_sequence(self, max_tokens: int) -> int:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        for i, (start, size) in enumerate(self.holes):
            if size >= max_tokens:
                sid = self._next_sequence
                self._next_sequence += 1
                self.allocations[sid] = (start, max_tokens, 0)
                if size == max_tokens:
                    self.holes.pop(i)
                else:
                    self.holes[i] = (start + max_tokens, size - max_tokens)
                return sid
        raise AllocationError("no sufficiently large contiguous extent")

    def reserve(self, sid: int, length: int) -> None:
        start, capacity, used = self.allocations[sid]
        if length < used:
            raise ValueError("reserve cannot shrink a sequence")
        if length > capacity:
            raise AllocationError("declared maximum length exceeded")
        self.allocations[sid] = (start, capacity, length)

    def free(self, sid: int) -> None:
        start, capacity, _ = self.allocations.pop(sid)
        holes = sorted(self.holes + [(start, capacity)])
        self.holes = []
        for lo, size in holes:
            if self.holes and self.holes[-1][0] + self.holes[-1][1] == lo:
                prev, prev_size = self.holes[-1]
                self.holes[-1] = (prev, prev_size + size)
            else:
                self.holes.append((lo, size))

    def metrics(self) -> dict:
        allocated = sum(n for _, n, _ in self.allocations.values())
        useful = sum(n for _, _, n in self.allocations.values())
        free = self.capacity - allocated
        largest = max((n for _, n in self.holes), default=0)
        return {
            "physical_capacity_tokens": self.capacity,
            "allocated_slots": allocated, "useful_slots": useful,
            "pool_occupancy": allocated / self.capacity,
            "allocated_utilization": useful / allocated if allocated else 0.0,
            "internal_fragmentation": (allocated - useful) / allocated if allocated else 0.0,
            "free_tokens": free, "largest_free_extent": largest,
            "external_fragmentation": 1 - largest / free if free else 0.0,
        }

    def check_invariants(self) -> None:
        spans = [(lo, n) for lo, n, _ in self.allocations.values()] + self.holes
        end = 0
        for start, n in sorted(spans):
            assert start == end and n > 0
            end = start + n
        assert end == self.capacity
        assert all(0 <= used <= n for _, n, used in self.allocations.values())
