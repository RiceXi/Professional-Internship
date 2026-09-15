"""CPU page contents used by the hardware-independent memory manager."""
from __future__ import annotations

from copy import deepcopy


class ListStorage:
    """Fixed-size physical pages. Values stand in for token KV vectors."""

    def __init__(self, num_frames: int, block_size: int, host_pages: int = 0):
        if num_frames <= 0 or block_size <= 0 or host_pages < 0:
            raise ValueError("num_frames and block_size must be positive")
        self.num_frames = num_frames
        self.block_size = block_size
        self.host_pages = host_pages
        self.host = [None] * host_pages
        self.frames = [[None] * block_size for _ in range(num_frames)]

    def clear(self, frame: int) -> None:
        self.frames[frame] = [None] * self.block_size

    def copy(self, src: int, dst: int) -> None:
        self.frames[dst] = deepcopy(self.frames[src])

    def write(self, frame: int, offset: int, value) -> None:
        self.frames[frame][offset] = deepcopy(value)

    def read(self, frame: int, offset: int):
        return deepcopy(self.frames[frame][offset])

    def store(self, frame: int, host: int) -> None:
        self.host[host] = deepcopy(self.frames[frame])

    def load(self, host: int, frame: int) -> None:
        self.frames[frame] = deepcopy(self.host[host])

    def drop_host(self, host: int) -> None:
        self.host[host] = None

    def exchange(self, frame: int, host: int) -> None:
        self.frames[frame], self.host[host] = self.host[host], self.frames[frame]
