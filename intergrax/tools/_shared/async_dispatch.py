# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Run async callables from sync catalog tool handlers."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import TypeVar

T = TypeVar("T")


def run_async(coro: Coroutine[object, object, T]) -> T:
    """Execute ``coro`` from a synchronous tool service function."""
    if asyncio.iscoroutine(coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        return loop.run_until_complete(coro)
    raise TypeError("run_async expects a coroutine object")
