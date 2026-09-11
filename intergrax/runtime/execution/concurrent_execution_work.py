# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Concurrent canonical Execution work submission (DS-COUNCIL-01).

Smallest Execution-owned primitive for parallel independent work units.
Council and other deliberation hosts consume this seam — concurrency ownership
remains in Execution, not strategy contracts.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

from intergrax.contracts.concurrent_execution_work import ConcurrentExecutionWorkPolicy
from intergrax.runtime.execution.execution_work_port import ExecutionWorkPort
from intergrax.runtime.execution.request import ExecutionRequest

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
ResultT = TypeVar("ResultT")


class ConcurrentExecutionWorkDisposition(str, Enum):
    """Per-unit outcome of resilient concurrent execution."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ConcurrentExecutionWorkOutcome(Generic[ResultT]):
    """Typed success/failure outcome for one concurrent work unit."""

    disposition: ConcurrentExecutionWorkDisposition
    result: ResultT | None
    error: Exception | None

    def __post_init__(self) -> None:
        if self.disposition is ConcurrentExecutionWorkDisposition.SUCCEEDED:
            if self.error is not None:
                raise ValueError("succeeded outcome must not carry error")
            if self.result is None:
                raise ValueError("succeeded outcome must carry result")
            return
        if self.disposition is ConcurrentExecutionWorkDisposition.FAILED:
            if self.error is None:
                raise ValueError("failed outcome must carry error")
            if self.result is not None:
                raise ValueError("failed outcome must not carry result")
            return
        raise ValueError(f"unsupported disposition: {self.disposition!r}")


def _validate_policy(policy: ConcurrentExecutionWorkPolicy) -> None:
    if type(policy) is not ConcurrentExecutionWorkPolicy:
        raise TypeError("policy must be ConcurrentExecutionWorkPolicy")


async def _run_bounded_strict(
    port: ExecutionWorkPort[InputT, OutputT, ResultT],
    requests: tuple[ExecutionRequest[InputT, OutputT], ...],
    *,
    policy: ConcurrentExecutionWorkPolicy,
) -> tuple[ResultT, ...]:
    count = len(requests)
    worker_count = min(policy.max_concurrency, count)
    results: list[ResultT | None] = [None] * count
    queue: asyncio.Queue[int] = asyncio.Queue()
    for index in range(count):
        queue.put_nowait(index)

    shutdown = asyncio.Event()
    terminal_failure: BaseException | None = None
    terminal_lock = asyncio.Lock()
    workers: list[asyncio.Task[None]] = []

    async def worker() -> None:
        nonlocal terminal_failure
        while not shutdown.is_set():
            try:
                index = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if shutdown.is_set():
                return
            try:
                results[index] = await port.execute(requests[index])
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                async with terminal_lock:
                    if terminal_failure is None:
                        terminal_failure = exc
                shutdown.set()
                current = asyncio.current_task()
                for task in workers:
                    if task is not current:
                        task.cancel()
                return

    workers = [asyncio.create_task(worker()) for _ in range(worker_count)]
    try:
        await asyncio.gather(*workers, return_exceptions=True)
    except asyncio.CancelledError:
        shutdown.set()
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        raise

    if terminal_failure is not None:
        raise terminal_failure

    ordered: list[ResultT] = []
    for item in results:
        if item is None:
            raise RuntimeError("bounded concurrent work did not produce all results")
        ordered.append(item)
    return tuple(ordered)


async def _run_bounded_resilient(
    port: ExecutionWorkPort[InputT, OutputT, ResultT],
    requests: tuple[ExecutionRequest[InputT, OutputT], ...],
    *,
    policy: ConcurrentExecutionWorkPolicy,
) -> tuple[ConcurrentExecutionWorkOutcome[ResultT], ...]:
    count = len(requests)
    worker_count = min(policy.max_concurrency, count)
    outcomes: list[ConcurrentExecutionWorkOutcome[ResultT] | None] = [None] * count
    queue: asyncio.Queue[int | None] = asyncio.Queue()
    for index in range(count):
        queue.put_nowait(index)

    async def worker() -> None:
        while True:
            index = await queue.get()
            if index is None:
                queue.task_done()
                return
            try:
                try:
                    result = await port.execute(requests[index])
                except Exception as exc:
                    outcomes[index] = ConcurrentExecutionWorkOutcome(
                        disposition=ConcurrentExecutionWorkDisposition.FAILED,
                        result=None,
                        error=exc,
                    )
                else:
                    outcomes[index] = ConcurrentExecutionWorkOutcome(
                        disposition=ConcurrentExecutionWorkDisposition.SUCCEEDED,
                        result=result,
                        error=None,
                    )
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(worker_count)]
    try:
        try:
            await queue.join()
        except asyncio.CancelledError:
            for task in workers:
                task.cancel()
            raise
    finally:
        for _ in range(worker_count):
            queue.put_nowait(None)
        await asyncio.gather(*workers, return_exceptions=True)

    ordered: list[ConcurrentExecutionWorkOutcome[ResultT]] = []
    for item in outcomes:
        if item is None:
            raise RuntimeError("bounded resilient concurrent work did not produce all outcomes")
        ordered.append(item)
    return tuple(ordered)


async def execute_concurrent_execution_work(
    port: ExecutionWorkPort[InputT, OutputT, ResultT],
    requests: tuple[ExecutionRequest[InputT, OutputT], ...],
    *,
    policy: ConcurrentExecutionWorkPolicy,
) -> tuple[ResultT, ...]:
    """Execute independent work units concurrently through one Execution work port."""
    if type(requests) is not tuple:
        raise TypeError("requests must be tuple")
    if len(requests) == 0:
        raise ValueError("requests must not be empty")
    _validate_policy(policy)
    return await _run_bounded_strict(port, requests, policy=policy)


async def execute_concurrent_execution_work_resilient(
    port: ExecutionWorkPort[InputT, OutputT, ResultT],
    requests: tuple[ExecutionRequest[InputT, OutputT], ...],
    *,
    policy: ConcurrentExecutionWorkPolicy,
) -> tuple[ConcurrentExecutionWorkOutcome[ResultT], ...]:
    """Execute independent work units concurrently; capture per-unit failures."""
    if type(requests) is not tuple:
        raise TypeError("requests must be tuple")
    if len(requests) == 0:
        raise ValueError("requests must not be empty")
    _validate_policy(policy)
    return await _run_bounded_resilient(port, requests, policy=policy)
