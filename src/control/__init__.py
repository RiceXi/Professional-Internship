"""Control plane: scheduling + engine orchestration."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__all__ = ["Engine", "Scheduler"]

if TYPE_CHECKING:
    from .engine import Engine
    from .scheduler import Scheduler

_LAZY = {
    "Engine": (".engine", "Engine"),
    "Scheduler": (".scheduler", "Scheduler"),
}


def __getattr__(name: str) -> Any:
    """懒加载：让调度器可以在没有 torch 的 CPU 环境单独导入和测试。"""
    try:
        module, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(importlib.import_module(module, __name__), attr)
