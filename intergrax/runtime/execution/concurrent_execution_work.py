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


async def execute_concurrent_execution_work(
    port: ExecutionWorkPort[InputT, OutputT, ResultT],
    requests: tuple[ExecutionRequest[InputT, OutputT], ...],
) -> tuple[ResultT, ...]:
    """Execute independent work units concurrently through one Execution work port."""
    if type(requests) is not tuple:
        raise TypeError("requests must be tuple")
    if len(requests) == 0:
        raise ValueError("requests must not be empty")
    results = await asyncio.gather(*(port.execute(request) for request in requests))
    return tuple(results)


async def _execute_one_resilient(
    port: ExecutionWorkPort[InputT, OutputT, ResultT],
    request: ExecutionRequest[InputT, OutputT],
) -> ConcurrentExecutionWorkOutcome[ResultT]:
    try:
        result = await port.execute(request)
    except Exception as exc:
        return ConcurrentExecutionWorkOutcome(
            disposition=ConcurrentExecutionWorkDisposition.FAILED,
            result=None,
            error=exc,
        )
    return ConcurrentExecutionWorkOutcome(
        disposition=ConcurrentExecutionWorkDisposition.SUCCEEDED,
        result=result,
        error=None,
    )


async def execute_concurrent_execution_work_resilient(
    port: ExecutionWorkPort[InputT, OutputT, ResultT],
    requests: tuple[ExecutionRequest[InputT, OutputT], ...],
) -> tuple[ConcurrentExecutionWorkOutcome[ResultT], ...]:
    """Execute independent work units concurrently; capture per-unit failures."""
    if type(requests) is not tuple:
        raise TypeError("requests must be tuple")
    if len(requests) == 0:
        raise ValueError("requests must not be empty")
    outcomes = await asyncio.gather(
        *(_execute_one_resilient(port, request) for request in requests),
    )
    return tuple(outcomes)
