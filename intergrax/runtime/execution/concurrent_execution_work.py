# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Concurrent canonical Execution work submission (DS-COUNCIL-01).

Smallest Execution-owned primitive for parallel independent work units.
Council and other deliberation hosts consume this seam — concurrency ownership
remains in Execution, not strategy contracts.
"""

from __future__ import annotations

import asyncio
from typing import TypeVar

from intergrax.runtime.execution.execution_work_port import ExecutionWorkPort
from intergrax.runtime.execution.request import ExecutionRequest

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
ResultT = TypeVar("ResultT")


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
