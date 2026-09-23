# © Artur Czarnecki. All rights reserved.

"""Protocol for declarative tool invokers that accept run-scoped binding."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.execution_bound_declarative_tool_invocation import (
    ExecutionBoundDeclarativeToolInvoker,
)


@runtime_checkable
class DeclarativeToolInvokerWithRunBinding(ExecutionBoundDeclarativeToolInvoker, Protocol):
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
