# © Artur Czarnecki. All rights reserved.

"""Bind declarative tool invoker to compensation execution identity (ACP)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.agents.persistence.catalog_declarative_invoker import (
    CatalogDeclarativeToolInvoker,
)
from intergrax.agents.persistence.declarative_tool_executor import DeclarativeToolInvoker
from intergrax.contracts.compensation_side_effect_execution import (
    CompensationSideEffectInvokeResult,
    CompensationToolInvokeSession,
)


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
        args: dict[str, Any],
        idempotency_key: str,
    ) -> CompensationSideEffectInvokeResult:
        if isinstance(self._invoker, CatalogDeclarativeToolInvoker):
            self._invoker.bind_run(
                run_id=run_id,
                task_id=task_id,
                agent_id=agent_id,
                tenant_id=tenant_id,
            )
        invoke_result = await self._invoker.invoke(
            tool_id=tool_id,
            args=args,
            idempotency_key=idempotency_key,
        )
        return CompensationSideEffectInvokeResult(
            status=invoke_result.status,
            error=invoke_result.error,
        )


def bound_compensation_tool_invoke_session(
    invoker: DeclarativeToolInvoker,
) -> BoundCompensationToolInvokeSession:
    return BoundCompensationToolInvokeSession(_invoker=invoker)
