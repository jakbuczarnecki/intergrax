# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from intergrax.contracts.idempotency_store import (
    ActiveInvocationClaimError,
    ClaimOutcome,
    ClaimResult,
    IdempotencyStore,
    InvocationClaim,
    InvocationUncertaintyError,
)
from intergrax.runtime.nexus.engine.contracts.runtime_state_contract import RuntimeStateContract
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel
from intergrax.runtime.tools.operation_identity import compute_invocation_operation_identity
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import (
    ToolEffectCertainty,
    ToolExecutionRequest,
    ToolExecutionResult,
)

if TYPE_CHECKING:
    from pydantic import BaseModel as PydanticBaseModel

_DEFAULT_LEASE_SECONDS = 300


def classify_idempotency_outcome(
    contract: ToolContract,
    result: ToolExecutionResult[BaseModel],
) -> Literal["safe_terminal", "uncertain"]:
    """Classify ledger terminal transition for a side-effect tool invocation."""
    if not contract.side_effects:
        return "safe_terminal"
    if result.success:
        return "safe_terminal"
    if result.effect_certainty == ToolEffectCertainty.NOT_STARTED:
        return "safe_terminal"
    return "uncertain"


class PreEffectClaimContext(BaseModel):
    """Request-scoped idempotency claim handle for one invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    key: str
    claim: InvocationClaim


class PreEffectCoordinationResult(BaseModel):
    """Typed pre-effect idempotency decision before external tool execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: ClaimOutcome
    claim_context: PreEffectClaimContext | None = None
    replay_result: ToolExecutionResult[BaseModel] | None = None


class IdempotencyPreEffectCoordinator:
    """
    Idempotency coordination immediately before external tool effects.

    Owns claim acquisition, replay disposition, and post-effect ledger finalization.
    Does not execute tools, evaluate governance, or persist claims on shared state.
    """

    def __init__(
        self,
        *,
        idempotency_store: IdempotencyStore,
        lease_seconds: int = _DEFAULT_LEASE_SECONDS,
    ) -> None:
        self._store = idempotency_store
        self._lease_seconds = lease_seconds

    def before_external_effect(
        self,
        *,
        state: RuntimeStateContract,
        contract: ToolContract,
        request: ToolExecutionRequest[PydanticBaseModel],
    ) -> PreEffectCoordinationResult:
        tenant_id = state.tenant_id
        key = request.idempotency_key
        if key is None:
            raise RuntimeError("Idempotency key is required for side-effect coordination.")
        owner_id = f"invoker-{uuid4().hex}"
        operation_identity = compute_invocation_operation_identity(
            request.tool_id,
            request.input,
        )

        claim_result = self._store.claim(
            tenant_id,
            key,
            owner_id,
            self._lease_seconds,
            operation_identity=operation_identity,
        )
        return self._to_coordination_result(
            state=state,
            contract=contract,
            request=request,
            claim_result=claim_result,
            tenant_id=tenant_id,
            key=key,
        )

    def after_external_effect(
        self,
        *,
        claim_context: PreEffectClaimContext,
        contract: ToolContract,
        result: ToolExecutionResult[BaseModel],
    ) -> None:
        outcome_kind = classify_idempotency_outcome(contract, result)
        if outcome_kind == "safe_terminal":
            self._store.complete_with_claim(
                claim_context.tenant_id,
                claim_context.key,
                claim_context.claim,
                result,
            )
        else:
            self._store.mark_uncertain_with_claim(
                claim_context.tenant_id,
                claim_context.key,
                claim_context.claim,
            )

    def _to_coordination_result(
        self,
        *,
        state: RuntimeStateContract,
        contract: ToolContract,
        request: ToolExecutionRequest[PydanticBaseModel],
        claim_result: ClaimResult,
        tenant_id: str,
        key: str,
    ) -> PreEffectCoordinationResult:
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
                    f"(tool_id={contract.tool_id}, key={key})."
                ),
            )
            return PreEffectCoordinationResult(
                outcome=ClaimOutcome.REPLAY_COMPLETED,
                replay_result=cached,
            )

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

        return PreEffectCoordinationResult(
            outcome=ClaimOutcome.ACQUIRED,
            claim_context=PreEffectClaimContext(
                tenant_id=tenant_id,
                key=key,
                claim=claim,
            ),
        )
