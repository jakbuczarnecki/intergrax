# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel

from intergrax.runtime.nexus.engine.contracts.runtime_state_contract import RuntimeStateContract
from intergrax.runtime.nexus.tracing.trace_models import (
    TraceComponent,
    TraceLevel,
)
from intergrax.contracts.idempotency_store import (
    ActiveInvocationClaimError,
    ClaimOutcome,
    IdempotencyStore,
    InvocationUncertaintyError,
)
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.execution_models import (
    ToolExecutionRequest,
    ToolExecutionResult,
)

if TYPE_CHECKING:
    from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker

_DEFAULT_LEASE_SECONDS = 300


class IdempotentToolInvoker:
    """
    Ledger-based idempotent tool invoker.

    Provides duplicate suppression and execution-uncertainty tracking for tools
    with side effects. Does not claim exactly-once across external boundaries.
    """

    def __init__(
        self,
        *,
        base_invoker: RuntimeToolInvoker,
        idempotency_store: IdempotencyStore,
        lease_seconds: int = _DEFAULT_LEASE_SECONDS,
    ) -> None:
        self._base_invoker = base_invoker
        self._store = idempotency_store
        self._lease_seconds = lease_seconds

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

        if not contract.side_effects or not request.idempotency_key:
            return self._base_invoker.invoke(
                state=state,
                agent_id=agent_id,
                request=request,
            )

        tenant_id = state.tenant_id
        key = request.idempotency_key
        owner_id = f"invoker-{uuid4().hex}"

        claim_result = self._store.claim(
            tenant_id,
            key,
            owner_id,
            self._lease_seconds,
        )

        if claim_result.outcome == ClaimOutcome.REPLAY_COMPLETED:
            cached = claim_result.completed_result
            if cached is None:
                cached = self._store.get_completed_result(tenant_id, key)
            if cached is None:
                raise RuntimeError(
                    "Ledger inconsistency: COMPLETED without stored result.",
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

        if claim_result.outcome == ClaimOutcome.BLOCKED_ACTIVE:
            raise ActiveInvocationClaimError(
                f"Invocation already claimed for key={key}. "
                "Blocking concurrent execution.",
            )

        if claim_result.outcome == ClaimOutcome.UNCERTAIN:
            raise InvocationUncertaintyError(
                f"Invocation outcome uncertain for key={key}. "
                "Reconciliation required before retry.",
            )

        claim = claim_result.claim
        if claim is None:
            raise RuntimeError("Ledger inconsistency: ACQUIRED without claim.")

        result = self._base_invoker.invoke(
            state=state,
            agent_id=agent_id,
            request=request,
        )

        self._store.complete_with_claim(tenant_id, key, claim, result)

        return result

    @staticmethod
    def docstring_denies_exactly_once() -> bool:
        """Inspection helper: invoker must not claim universal exactly-once."""
        doc = IdempotentToolInvoker.__doc__ or ""
        return "enforces exactly-once" not in doc.lower()
