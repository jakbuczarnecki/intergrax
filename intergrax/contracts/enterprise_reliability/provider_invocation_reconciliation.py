# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider invocation reconciliation — durable UNKNOWN probe eligibility (GR-7-A6)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.enterprise_reliability.effect_contract import (
    ExternalEffectContract,
    contract_declares_reconciliation,
)
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
    ProviderInvocationStatus,
)

_MAX_REASON = 512


class ProviderInvocationReconciliationVerdict(StrEnum):
    """Typed reconciliation truth for one durable provider attempt."""

    NOT_EXECUTED = "not_executed"
    NOT_AVAILABLE = "not_available"
    PROBE_FAILED = "probe_failed"
    CONFIRMED_SUCCEEDED = "confirmed_succeeded"
    CONFIRMED_FAILED = "confirmed_failed"
    STILL_UNKNOWN = "still_unknown"


class ProviderInvocationReconciliationReason(StrEnum):
    """Closed validation and scheduling rationale."""

    PROBE_EXECUTED = "probe_executed"
    RECONCILIATION_UNSUPPORTED = "reconciliation_unsupported"
    ESCALATE_REQUIRED = "escalate_required"
    DENIED_INVALID_STATE = "denied_invalid_state"
    DENIED_OUTCOME_MISSING = "denied_outcome_missing"
    DENIED_OUTCOME_STATUS = "denied_outcome_status"
    DENIED_INVOCATION_MISSING = "denied_invocation_missing"
    DENIED_CORRELATION_INFEASIBLE = "denied_correlation_infeasible"
    PLUGIN_PROBE_UNAVAILABLE = "plugin_probe_unavailable"
    SKIPPED_NOT_SCHEDULED = "skipped_not_scheduled"


class ProviderInvocationReconciliationRequest(BaseModel):
    """Canonical reconciliation inputs — no loose task_id-only requests."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    invocation: ProviderInvocation | None = None
    outcome: ProviderInvocationOutcome | None = None
    effect_contract: ExternalEffectContract
    tenant_id: str = Field(min_length=1, max_length=256)
    plugin_id: str = Field(min_length=1, max_length=256)
    provider_id: str | None = Field(default=None, max_length=256)


class ProviderInvocationReconciliationPreparation(BaseModel):
    """Validated request ready for ERL reconcile orchestration — still no provider I/O."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    invocation: ProviderInvocation
    outcome: ProviderInvocationOutcome
    effect_contract: ExternalEffectContract
    correlation_id: str
    tenant_id: str
    plugin_id: str


class ProviderInvocationReconciliationResult(BaseModel):
    """One reconciliation attempt bound to invocation identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: ProviderInvocationReconciliationVerdict
    reason: ProviderInvocationReconciliationReason
    invocation_id: str | None = None
    evidence_ref: str | None = None
    detail: str = Field(default="", max_length=_MAX_REASON)


def provider_invocation_reconciliation_correlation_id(
    invocation: ProviderInvocation,
) -> str:
    """Align ERL correlation with governed external-work admission."""
    if invocation.correlation_id and invocation.correlation_id.strip():
        return invocation.correlation_id.strip()
    if invocation.idempotency_key and invocation.idempotency_key.strip():
        return invocation.idempotency_key.strip()
    return invocation.task_id.strip()


def prepare_provider_invocation_reconciliation(
    request: ProviderInvocationReconciliationRequest,
) -> ProviderInvocationReconciliationPreparation | ProviderInvocationReconciliationResult:
    """
    Fail-closed validation before any reconcile planning or provider read.

    Returns a preparation on success, or a typed NOT_EXECUTED / NOT_AVAILABLE result.
    """
    invocation = request.invocation
    if invocation is None:
        return ProviderInvocationReconciliationResult(
            verdict=ProviderInvocationReconciliationVerdict.NOT_EXECUTED,
            reason=ProviderInvocationReconciliationReason.DENIED_INVOCATION_MISSING,
            detail="provider invocation required",
        )

    outcome = request.outcome
    if outcome is None:
        return ProviderInvocationReconciliationResult(
            verdict=ProviderInvocationReconciliationVerdict.NOT_EXECUTED,
            reason=ProviderInvocationReconciliationReason.DENIED_OUTCOME_MISSING,
            invocation_id=invocation.invocation_id,
            detail="provider outcome required",
        )

    if outcome.invocation_id != invocation.invocation_id:
        return ProviderInvocationReconciliationResult(
            verdict=ProviderInvocationReconciliationVerdict.NOT_EXECUTED,
            reason=ProviderInvocationReconciliationReason.DENIED_INVALID_STATE,
            invocation_id=invocation.invocation_id,
            detail="outcome invocation_id mismatch",
        )

    contract = request.effect_contract
    if contract.operation_key != invocation.operation:
        return ProviderInvocationReconciliationResult(
            verdict=ProviderInvocationReconciliationVerdict.NOT_EXECUTED,
            reason=ProviderInvocationReconciliationReason.DENIED_INVALID_STATE,
            invocation_id=invocation.invocation_id,
            detail="effect contract operation_key mismatch",
        )

    bound_provider = request.provider_id
    if bound_provider is not None and bound_provider.strip():
        if bound_provider.strip() != invocation.provider_id:
            return ProviderInvocationReconciliationResult(
                verdict=ProviderInvocationReconciliationVerdict.NOT_EXECUTED,
                reason=ProviderInvocationReconciliationReason.DENIED_INVALID_STATE,
                invocation_id=invocation.invocation_id,
                detail="provider_id binding mismatch",
            )

    if outcome.status is not ProviderInvocationStatus.UNKNOWN:
        return ProviderInvocationReconciliationResult(
            verdict=ProviderInvocationReconciliationVerdict.NOT_EXECUTED,
            reason=ProviderInvocationReconciliationReason.DENIED_OUTCOME_STATUS,
            invocation_id=invocation.invocation_id,
            detail=f"outcome status {outcome.status.value} is not UNKNOWN",
        )

    if not contract_declares_reconciliation(contract):
        return ProviderInvocationReconciliationResult(
            verdict=ProviderInvocationReconciliationVerdict.NOT_AVAILABLE,
            reason=ProviderInvocationReconciliationReason.RECONCILIATION_UNSUPPORTED,
            invocation_id=invocation.invocation_id,
            detail="effective contract reconciliation not supported",
        )

    external_task_id = invocation.external_task_id
    if external_task_id is None or not external_task_id.strip():
        return ProviderInvocationReconciliationResult(
            verdict=ProviderInvocationReconciliationVerdict.NOT_AVAILABLE,
            reason=ProviderInvocationReconciliationReason.DENIED_CORRELATION_INFEASIBLE,
            invocation_id=invocation.invocation_id,
            detail="reconciliation probe requires durable external_task_id correlation",
        )

    return ProviderInvocationReconciliationPreparation(
        invocation=invocation,
        outcome=outcome,
        effect_contract=contract,
        correlation_id=provider_invocation_reconciliation_correlation_id(invocation),
        tenant_id=request.tenant_id.strip(),
        plugin_id=request.plugin_id.strip(),
    )


__all__ = [
    "ProviderInvocationReconciliationPreparation",
    "ProviderInvocationReconciliationReason",
    "ProviderInvocationReconciliationRequest",
    "ProviderInvocationReconciliationResult",
    "ProviderInvocationReconciliationVerdict",
    "prepare_provider_invocation_reconciliation",
    "provider_invocation_reconciliation_correlation_id",
]
