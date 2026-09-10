# © Artur Czarnecki. All rights reserved.

"""Bind declarative tool invoker to compensation execution identity (ACP)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.agents.persistence.declarative_tool_executor import (
    DeclarativeToolInvoker,
    DeclarativeToolInvokeResult,
)
from intergrax.contracts.compensation_side_effect_execution import (
    CompensationSideEffectInvokeResult,
    CompensationToolInvokeSession,
)
from intergrax.contracts.execution_bound_declarative_tool_invocation import (
    ExecutionBoundDeclarativeToolInvoker,
)
from intergrax.knowledge.contracts.validation import JsonObject


@dataclass(frozen=True, slots=True)
class BoundCompensationToolInvokeSession(CompensationToolInvokeSession):
    _invoker: DeclarativeToolInvoker

    async def invoke(
        self,
        *,
        tenant_id: str,
        run_id: str,
        task_id: str,
        agent_id: str,
        tool_id: str,
        args: JsonObject,
        idempotency_key: str,
    ) -> CompensationSideEffectInvokeResult:
        if isinstance(self._invoker, ExecutionBoundDeclarativeToolInvoker):
            self._invoker.bind_execution_identity(
                tenant_id=tenant_id,
                run_id=run_id,
                task_id=task_id,
                agent_id=agent_id,
            )
        invoke_result = await self._invoker.invoke(
            tool_id=tool_id,
            args=dict(args),
            idempotency_key=idempotency_key,
        )
        if not isinstance(invoke_result, DeclarativeToolInvokeResult):
            raise TypeError("declarative tool invoker returned unexpected result type")
        return CompensationSideEffectInvokeResult(
            status=invoke_result.status,
            error=invoke_result.error,
        )


def bound_compensation_tool_invoke_session(
    invoker: DeclarativeToolInvoker,
) -> BoundCompensationToolInvokeSession:
    return BoundCompensationToolInvokeSession(_invoker=invoker)
