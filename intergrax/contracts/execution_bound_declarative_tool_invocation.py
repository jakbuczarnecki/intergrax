# © Artur Czarnecki. All rights reserved.

"""Execution-bound declarative tool invocation port (catalog / compensation U2)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.declarative_tool_invoke_result import DeclarativeToolInvokeResult
from intergrax.knowledge.contracts.validation import JsonObject


@runtime_checkable
class ExecutionBoundDeclarativeToolInvoker(Protocol):
    """Declarative invoker that must be rebound to active execution identity before invoke."""

    def bind_execution_identity(
        self,
        *,
        tenant_id: str,
        run_id: str,
        task_id: str,
        agent_id: str,
    ) -> None:
        ...

    async def invoke(
        self,
        *,
        tool_id: str,
        args: JsonObject,
        idempotency_key: str | None,
    ) -> DeclarativeToolInvokeResult:
        ...
