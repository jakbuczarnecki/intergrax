# © Artur Czarnecki. All rights reserved.

"""Execution-bound declarative tool invocation port (catalog / compensation U2)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.declarative_tool_invoke_result import DeclarativeToolInvokeResult
from intergrax.knowledge.contracts.validation import JsonObject


@runtime_checkable
class ExecutionBoundDeclarativeToolInvoker(Protocol):
    """Declarative invoker under an immutable per-call execution identity (ADR3 M3)."""

    async def invoke(
        self,
        *,
        tenant_id: str,
        run_id: str,
        task_id: str,
        agent_id: str,
        tool_id: str,
        args: JsonObject,
        idempotency_key: str | None,
    ) -> DeclarativeToolInvokeResult:
        ...
