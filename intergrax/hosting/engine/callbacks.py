# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared callback invocation semantics for hosting coordinators (APP-HOST-W2)."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")


def callback_invocation_kind(callback: Callable[..., Any]) -> str:
    """Classify callback invocation without calling it."""
    if inspect.iscoroutinefunction(callback):
        return "async"
    return "sync"


async def invoke_callback(callback: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Invoke a callback exactly once with correct sync/async semantics.

    - async callback: invoke once on the event loop, await once
    - ordinary sync callback: invoke once via ``asyncio.to_thread``, never on the loop
    - sync callback returning an awaitable: invoke once via thread, await result once
    """
    if inspect.iscoroutinefunction(callback):
        return await callback(*args, **kwargs)

    result = await asyncio.to_thread(callback, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


async def invoke_callback_awaitable(
    awaitable: Awaitable[T],
) -> T:
    """Await an already-produced awaitable exactly once."""
    return await awaitable
