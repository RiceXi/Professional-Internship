"""Data plane: KV cache, sequences, batch buffers, prefix cache."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .scheduled_batch import ScheduledBatch
from .sequence import Sequence, SequenceStatus

__all__ = ["Batch", "ScheduledBatch", "KVManager", "Sequence", "SequenceStatus"]

if TYPE_CHECKING:
    from .batch import Batch
    from .manager import KVManager

_LAZY = {
    "Batch": (".batch", "Batch"),
    "KVManager": (".manager", "KVManager"),
}


def __getattr__(name: str) -> Any:
    """懒加载张量相关对象，使序列与调度逻辑可在纯 CPU 环境导入。"""
    try:
        module, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(importlib.import_module(module, __name__), attr)
