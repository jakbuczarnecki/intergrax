# © Artur Czarnecki. All rights reserved.

"""Bounded real concurrency helpers for MEM-ENT-14-R (test-only)."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class WriterResult[T]:
    value: T | None = None
    error: Exception | None = None


def run_two_thread_writers(
    *,
    writer_a: Callable[[], T],
    writer_b: Callable[[], T],
    join_timeout: float = 5.0,
) -> tuple[WriterResult[T], WriterResult[T]]:
    """Run two writers on separate threads; propagate exceptions into typed results."""
    result_a: WriterResult[T] = WriterResult()
    result_b: WriterResult[T] = WriterResult()

    def _run(target: Callable[[], T], bucket: WriterResult[T]) -> None:
        try:
            bucket.value = target()
        except Exception as exc:
            bucket.error = exc

    thread_a = threading.Thread(target=_run, args=(writer_a, result_a))
    thread_b = threading.Thread(target=_run, args=(writer_b, result_b))
    thread_a.start()
    thread_b.start()
    thread_a.join(join_timeout)
    thread_b.join(join_timeout)
    if thread_a.is_alive() or thread_b.is_alive():
        raise TimeoutError("thread writers did not finish within join_timeout")
    return result_a, result_b


def run_two_writers_with_start_barrier(
    *,
    start_barrier: threading.Barrier,
    writer_a: Callable[[], T],
    writer_b: Callable[[], T],
    join_timeout: float = 5.0,
) -> tuple[WriterResult[T], WriterResult[T]]:
    """Release both writers simultaneously via a start ``threading.Barrier``."""

    def _gated(call: Callable[[], T]) -> T:
        start_barrier.wait(timeout=join_timeout)
        return call()

    return run_two_thread_writers(
        writer_a=lambda: _gated(writer_a),
        writer_b=lambda: _gated(writer_b),
        join_timeout=join_timeout,
    )


async def run_two_async_tasks(
    *,
    task_a: Callable[[], Awaitable[T]],
    task_b: Callable[[], Awaitable[T]],
) -> tuple[T, T]:
    return await asyncio.gather(task_a(), task_b())


async def run_two_async_tasks_with_barrier(
    *,
    start_barrier: asyncio.Barrier,
    task_a: Callable[[], Awaitable[T]],
    task_b: Callable[[], Awaitable[T]],
) -> tuple[T, T]:
    async def _gated(call: Callable[[], Awaitable[T]]) -> T:
        await start_barrier.wait()
        return await call()

    return await run_two_async_tasks(
        task_a=lambda: _gated(task_a),
        task_b=lambda: _gated(task_b),
    )
