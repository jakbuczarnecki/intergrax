# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bounded concurrent MCP tool invocations on the composition engine (MCP-P3-06)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TypeVar

from intergrax.applications._shared.strict_host_dependency_concurrency_admission import (
    REFERENCE_STRICT_PRODUCT_TOOL_MAX_CONCURRENT_CALLS,
)

DEFAULT_MCP_MAX_CONCURRENT_TOOL_CALLS: int = REFERENCE_STRICT_PRODUCT_TOOL_MAX_CONCURRENT_CALLS

_T = TypeVar("_T")


@asynccontextmanager
async def optional_mcp_call_admission(
    admission: McpEngineConcurrentCallAdmission | None,
) -> AsyncIterator[None]:
    """Enter concurrent-call admission when configured."""
    if admission is None:
        yield
        return
    async with admission:
        yield


class McpConcurrentToolCallLimitExceeded(RuntimeError):
    """Raised when in-flight MCP tool calls exceed configured admission (protocol-visible failure)."""

    def __init__(self, code: str = "mcp_concurrent_call_limit_exceeded") -> None:
        self.code = code
        super().__init__(code)


class McpEngineConcurrentCallAdmission:
    """Process-local admission for concurrent MCP tool handler executions."""

    def __init__(self, max_concurrent_calls: int) -> None:
        if max_concurrent_calls < 1:
            raise ValueError("max_concurrent_calls must be >= 1")
        self._max = max_concurrent_calls
        self._inflight = 0
        self._lock = asyncio.Lock()

    @property
    def max_concurrent_calls(self) -> int:
        return self._max

    async def __aenter__(self) -> None:
        async with self._lock:
            if self._inflight >= self._max:
                raise McpConcurrentToolCallLimitExceeded()
            self._inflight += 1

    async def __aexit__(self, *exc: object) -> None:
        async with self._lock:
            self._inflight -= 1


__all__ = [
    "DEFAULT_MCP_MAX_CONCURRENT_TOOL_CALLS",
    "McpConcurrentToolCallLimitExceeded",
    "McpEngineConcurrentCallAdmission",
    "optional_mcp_call_admission",
]
