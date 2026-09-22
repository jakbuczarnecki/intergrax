# © Artur Czarnecki. All rights reserved.

"""Protocol for declarative tool invokers that accept run-scoped binding."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.agents.persistence.declarative_tool_executor import DeclarativeToolInvoker


@runtime_checkable
class DeclarativeToolInvokerWithRunBinding(DeclarativeToolInvoker, Protocol):
    """Session metadata binding only; execution identity is per-call on ``invoke`` (ADR3 M4)."""

    def bind_run(
        self,
        *,
        run_id: str = "",
        task_id: str = "",
        agent_id: str = "",
        tenant_id: str = "",
        user_id: str = "",
    ) -> None: ...


@runtime_checkable
class PerCallExecutionIdentityDeclarativeToolInvoker(Protocol):
    async def invoke(
        self,
        *,
        tenant_id: str,
        run_id: str,
        task_id: str,
        agent_id: str,
        tool_id: str,
        args: dict[str, object],
        idempotency_key: str | None,
    ): ...
