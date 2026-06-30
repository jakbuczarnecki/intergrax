# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from intergrax.runtime.nexus.engine.contracts.runtime_state_contract import RuntimeStateContract
from intergrax.runtime.nexus.tracing.trace_models import (
    TraceComponent,
    TraceLevel,
)
from intergrax.contracts.idempotency_store import (
    IdempotencyStore,
    InvocationStatus,
)
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.execution_models import (
    ToolExecutionRequest,
    ToolExecutionResult,
)

if TYPE_CHECKING:
    from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker


class IdempotentToolInvoker:
    """
    Ledger-based idempotent tool invoker.

    Enforces exactly-once semantics for tools with side effects.
    """

    def __init__(
        self,
        *,
        base_invoker: RuntimeToolInvoker,
        idempotency_store: IdempotencyStore,
    ) -> None:
        self._base_invoker = base_invoker
        self._store = idempotency_store

    @property
    def registry(self) -> ToolRegistry:
        return self._base_invoker.registry

    def invoke(
        self,
        *,
        state: RuntimeStateContract,
        agent_id: str,
        request: ToolExecutionRequest[BaseModel],
    ) -> ToolExecutionResult[BaseModel]:

        registry = self._base_invoker.registry
        reg = registry.get(request.tool_id)
        contract = reg.contract

        # Non-side-effect tools → delegate directly
        if not contract.side_effects or not request.idempotency_key:
            return self._base_invoker.invoke(
                state=state,
                agent_id=agent_id,
                request=request,
            )

        tenant_id = state.tenant_id
        key = request.idempotency_key

        status = self._store.get_status(tenant_id, key)

        if status == InvocationStatus.COMPLETED:
            cached = self._store.get_completed_result(tenant_id, key)
            if cached is None:
                raise RuntimeError(
                    "Ledger inconsistency: COMPLETED without stored result."
                )

            state.trace_event(
                component=TraceComponent.TOOLS,
                step="idempotency_cache_hit",
                level=TraceLevel.INFO,
                message=(
                    f"Tool call deduplicated via idempotency "
                    f"(tool_id={request.tool_id}, key={key})."
                ),
            )

            return cached

        if status == InvocationStatus.STARTED:
            raise RuntimeError(
                f"Invocation already started for key={key}. "
                "Blocking to preserve exactly-once semantics."
            )

        # NONE → transition to STARTED
        self._store.record_started(tenant_id, key)

        result = self._base_invoker.invoke(
            state=state,
            agent_id=agent_id,
            request=request,
        )

        # Transition to COMPLETED
        self._store.record_completed(tenant_id, key, result)

        return result
