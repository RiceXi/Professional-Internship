"""Scheduler output type, importable without the tensor runtime."""

from __future__ import annotations

from dataclasses import dataclass, field

from .sequence import Sequence


@dataclass
class ScheduledBatch:
    """调度器产出的本步批次。compute 层只消费这个显式接口，不触碰 Sequence 内部。"""

    prefills: list[Sequence] = field(default_factory=list)
    decodes: list[Sequence] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.prefills and not self.decodes
