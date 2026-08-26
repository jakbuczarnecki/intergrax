# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical execution boundary coordination skeleton (UE-1B)."""

from __future__ import annotations

from typing import Generic, Protocol, TypeVar

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


class ExecutionDelegate(Protocol[RequestT, ResultT]):
    """Typed executor invoked exactly once by :class:`ExecutionBoundary`."""

    async def execute(self, request: RequestT) -> ResultT:
        ...


class ExecutionBoundary(Generic[RequestT, ResultT]):
    """
    Canonical coordination boundary skeleton (UE-1B).

    Receives a typed request, invokes a typed delegate once, and returns the
    typed result unchanged. Does not own subsystem semantics (policy, budget,
    observability, checkpoint, strategy selection, or executor internals).
    Strategy-neutral. Production routing is unchanged at UE-1B.
    """

    __slots__ = ("_delegate",)

    def __init__(self, delegate: ExecutionDelegate[RequestT, ResultT]) -> None:
        self._delegate = delegate

    async def execute(self, request: RequestT) -> ResultT:
        return await self._delegate.execute(request)
