# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Developer-facing execution facade (UE-1C)."""

from __future__ import annotations

from typing import Generic, TypeVar

from intergrax.runtime.execution.boundary import ExecutionBoundary

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


class Execution(Generic[RequestT, ResultT]):
    """
    Typed developer-facing facade establishing ``await execution.execute(request)``.

    Wraps an existing :class:`ExecutionBoundary` explicitly supplied at construction.
    Does not own subsystem semantics, strategy selection, or global runtime state.
    """

    __slots__ = ("_boundary",)

    def __init__(self, boundary: ExecutionBoundary[RequestT, ResultT]) -> None:
        self._boundary = boundary

    async def execute(self, request: RequestT) -> ResultT:
        return await self._boundary.execute(request)
