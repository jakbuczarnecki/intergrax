# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Developer-facing execution facade (UE-1C, UE-10R1)."""

from __future__ import annotations

from typing import Generic, TypeVar

from intergrax.runtime.execution.runtime import (
    ExecutionRuntime,
    RootExecutionOptions,
    resolve_root_execution_context,
)

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


class Execution(Generic[RequestT, ResultT]):
    """
    Typed developer-facing facade establishing canonical root execution.

    Wraps :class:`ExecutionRuntime` supplied at construction. Does not own
    subsystem semantics, strategy selection, or global runtime state.
    """

    __slots__ = ("_runtime",)

    def __init__(self, runtime: ExecutionRuntime[RequestT, ResultT]) -> None:
        self._runtime = runtime

    async def execute(
        self,
        request: RequestT,
        *,
        options: RootExecutionOptions,
    ) -> ResultT:
        root_context = resolve_root_execution_context(options)
        return await self._runtime.execute(request, root_context)
