# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider invocation recovery execution — approved action only (GR-7-A7)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.enterprise_reliability.effect_contract import ExternalEffectContract
from intergrax.contracts.enterprise_reliability.plugin_spi import EnterpriseReliabilityPluginGateway
from intergrax.contracts.enterprise_reliability.provider_invocation_reconciliation import (
    ProviderInvocationReconciliationRequest,
)
from intergrax.contracts.enterprise_reliability.provider_invocation_recovery import (
    ProviderInvocationRecoveryAction,
    ProviderInvocationRecoveryDecision,
    ProviderInvocationRecoveryEscalationContext,
    ProviderInvocationRecoveryPolicy,
    ProviderInvocationRecoveryRequest,
    evaluate_provider_invocation_recovery,
)
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
)
from intergrax.runtime.enterprise_reliability.provider_invocation_reconciliation import (
    ProviderInvocationReconciliationRun,
    reconcile_durable_provider_invocation_unknown,
)
from intergrax.runtime.enterprise_reliability.provider_invocation_recovery_execution_validation import (
    ProviderInvocationRecoveryExecutionBlockReason,
    validate_idempotent_repeat_port_result,
    validate_provider_invocation_recovery_execution,
)

_MAX_REASON = 512


class ProviderInvocationRecoveryExecutionDisposition(StrEnum):
    NOT_ATTEMPTED = "not_attempted"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class ProviderInvocationRecoveryRepeatResult(BaseModel):
    """Outcome of one idempotent provider repeat — new physical invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repeat_invocation_id: str = Field(min_length=1, max_length=256)
    idempotency_key: str = Field(min_length=1, max_length=512)
    provider_mutation_count: int = Field(ge=0, le=1)


class ProviderInvocationRecoveryHitlResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    escalation: ProviderInvocationRecoveryEscalationContext
    governed_continuation_request_id: str | None = Field(default=None, max_length=256)


class ProviderInvocationRecoveryExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: ProviderInvocationRecoveryDecision
    execution_attempted: bool
    disposition: ProviderInvocationRecoveryExecutionDisposition
    provider_mutation_count: int = Field(ge=0)
    reconciliation: ProviderInvocationReconciliationRun | None = None
    repeat: ProviderInvocationRecoveryRepeatResult | None = None
    hitl: ProviderInvocationRecoveryHitlResult | None = None
    block_reason: ProviderInvocationRecoveryExecutionBlockReason | None = None
    detail: str = Field(default="", max_length=_MAX_REASON)


class ProviderInvocationRecoveryRepeatPort(Protocol):
    """Host-owned repeat — must use dispatch gate and GR-6 authorization path."""

    def supports_provider_operation(self, operation: str) -> bool:
        """Whether this executor can perform idempotent repeat for ``operation``."""

    def execute_idempotent_repeat(
        self,
        *,
        original_invocation: ProviderInvocation,
        original_outcome: ProviderInvocationOutcome,
        effect_contract: ExternalEffectContract,
    ) -> ProviderInvocationRecoveryRepeatResult:
        """Exactly one provider mutation with same idempotency identity."""


class ProviderInvocationRecoveryHitlPort(Protocol):
    def surface_hitl(
        self,
        escalation: ProviderInvocationRecoveryEscalationContext,
    ) -> ProviderInvocationRecoveryHitlResult:
        """Zero provider mutations — existing continuation/HITL composition."""


@dataclass(frozen=True, slots=True)
class ProviderInvocationRecoveryExecutionPorts:
    gateway: EnterpriseReliabilityPluginGateway | None = None
    repeat: ProviderInvocationRecoveryRepeatPort | None = None
    hitl: ProviderInvocationRecoveryHitlPort | None = None


def decide_provider_invocation_recovery(
    request: ProviderInvocationRecoveryRequest,
    *,
    policy: ProviderInvocationRecoveryPolicy | None = None,
) -> ProviderInvocationRecoveryDecision:
    return evaluate_provider_invocation_recovery(request, policy=policy)


def build_recovery_escalation_context(
    *,
    request: ProviderInvocationRecoveryRequest,
    decision: ProviderInvocationRecoveryDecision,
) -> ProviderInvocationRecoveryEscalationContext | None:
    invocation = request.invocation
    if invocation is None:
        return None
    reconciliation = request.reconciliation
    repeat = request.repeat_eligibility
    return ProviderInvocationRecoveryEscalationContext(
        invocation=invocation,
        outcome=request.outcome,
        dispatch_state=request.dispatch_state,
        effect_contract_id=request.effect_contract.contract_id,
        recovery_reason=decision.reason,
        reconciliation_invocation_id=(
            reconciliation.invocation_id if reconciliation is not None else None
        ),
        reconciliation_verdict=(
            reconciliation.verdict if reconciliation is not None else None
        ),
        repeat_eligibility_verdict=(
            repeat.verdict if repeat is not None else None
        ),
        repeat_eligibility_reason=(
            repeat.reason.value if repeat is not None else None
        ),
        detail=decision.detail,
    )


def execute_provider_invocation_recovery(
    request: ProviderInvocationRecoveryRequest,
    *,
    decision: ProviderInvocationRecoveryDecision,
    ports: ProviderInvocationRecoveryExecutionPorts,
    reconciliation_request: ProviderInvocationReconciliationRequest | None = None,
) -> ProviderInvocationRecoveryExecutionResult:
    """
    Execute at most one recovery action for a prior decision evaluation.

    Never falls through to a second action on failure.
    """
    action = decision.action
    if action in {
        ProviderInvocationRecoveryAction.NO_ACTION,
        ProviderInvocationRecoveryAction.TERMINAL_SUCCESS,
        ProviderInvocationRecoveryAction.TERMINAL_FAILURE,
    }:
        return ProviderInvocationRecoveryExecutionResult(
            decision=decision,
            execution_attempted=False,
            disposition=ProviderInvocationRecoveryExecutionDisposition.NOT_ATTEMPTED,
            provider_mutation_count=0,
        )

    if action is ProviderInvocationRecoveryAction.RECONCILE:
        binding = validate_provider_invocation_recovery_execution(
            request,
            decision,
            expected_action=ProviderInvocationRecoveryAction.RECONCILE,
        )
        if not binding.allowed:
            return ProviderInvocationRecoveryExecutionResult(
                decision=decision,
                execution_attempted=True,
                disposition=ProviderInvocationRecoveryExecutionDisposition.BLOCKED,
                provider_mutation_count=0,
                block_reason=binding.block_reason,
            )
        if ports.gateway is None or reconciliation_request is None:
            return ProviderInvocationRecoveryExecutionResult(
                decision=decision,
                execution_attempted=True,
                disposition=ProviderInvocationRecoveryExecutionDisposition.BLOCKED,
                provider_mutation_count=0,
                detail="reconciliation ports incomplete",
            )
        try:
            recon_run = reconcile_durable_provider_invocation_unknown(
                reconciliation_request,
                gateway=ports.gateway,
            )
        except Exception as exc:
            return ProviderInvocationRecoveryExecutionResult(
                decision=decision,
                execution_attempted=True,
                disposition=ProviderInvocationRecoveryExecutionDisposition.FAILED,
                provider_mutation_count=0,
                detail=str(exc)[:_MAX_REASON],
            )
        return ProviderInvocationRecoveryExecutionResult(
            decision=decision,
            execution_attempted=True,
            disposition=ProviderInvocationRecoveryExecutionDisposition.COMPLETED,
            provider_mutation_count=0,
            reconciliation=recon_run,
        )

    if action is ProviderInvocationRecoveryAction.IDEMPOTENT_REPEAT:
        validation = validate_provider_invocation_recovery_execution(
            request,
            decision,
            expected_action=ProviderInvocationRecoveryAction.IDEMPOTENT_REPEAT,
        )
        if not validation.allowed:
            return ProviderInvocationRecoveryExecutionResult(
                decision=decision,
                execution_attempted=True,
                disposition=ProviderInvocationRecoveryExecutionDisposition.BLOCKED,
                provider_mutation_count=0,
                block_reason=validation.block_reason,
            )
        if ports.repeat is None:
            return ProviderInvocationRecoveryExecutionResult(
                decision=decision,
                execution_attempted=True,
                disposition=ProviderInvocationRecoveryExecutionDisposition.BLOCKED,
                provider_mutation_count=0,
                detail="repeat port unavailable",
            )
        if not request.repeat_execution_supported:
            return ProviderInvocationRecoveryExecutionResult(
                decision=decision,
                execution_attempted=True,
                disposition=ProviderInvocationRecoveryExecutionDisposition.BLOCKED,
                provider_mutation_count=0,
                block_reason=ProviderInvocationRecoveryExecutionBlockReason.REPEAT_EXECUTION_UNSUPPORTED,
            )
        invocation = request.invocation
        outcome = request.outcome
        if invocation is None or outcome is None:
            return ProviderInvocationRecoveryExecutionResult(
                decision=decision,
                execution_attempted=True,
                disposition=ProviderInvocationRecoveryExecutionDisposition.BLOCKED,
                provider_mutation_count=0,
                block_reason=ProviderInvocationRecoveryExecutionBlockReason.INVOCATION_MISSING,
            )
        invocation_key = (invocation.idempotency_key or "").strip()
        try:
            repeat_result = ports.repeat.execute_idempotent_repeat(
                original_invocation=invocation,
                original_outcome=outcome,
                effect_contract=request.effect_contract,
            )
        except Exception as exc:
            return ProviderInvocationRecoveryExecutionResult(
                decision=decision,
                execution_attempted=True,
                disposition=ProviderInvocationRecoveryExecutionDisposition.FAILED,
                provider_mutation_count=0,
                detail=str(exc)[:_MAX_REASON],
            )
        post_validation = validate_idempotent_repeat_port_result(
            original_invocation_id=invocation.invocation_id,
            original_idempotency_key=invocation_key,
            repeat_invocation_id=repeat_result.repeat_invocation_id,
            repeat_idempotency_key=repeat_result.idempotency_key,
            provider_mutation_count=repeat_result.provider_mutation_count,
        )
        if not post_validation.allowed:
            return ProviderInvocationRecoveryExecutionResult(
                decision=decision,
                execution_attempted=True,
                disposition=ProviderInvocationRecoveryExecutionDisposition.FAILED,
                provider_mutation_count=repeat_result.provider_mutation_count,
                repeat=repeat_result,
                block_reason=post_validation.block_reason,
            )
        disposition = ProviderInvocationRecoveryExecutionDisposition.COMPLETED
        if repeat_result.provider_mutation_count == 0:
            disposition = ProviderInvocationRecoveryExecutionDisposition.BLOCKED
        return ProviderInvocationRecoveryExecutionResult(
            decision=decision,
            execution_attempted=True,
            disposition=disposition,
            provider_mutation_count=repeat_result.provider_mutation_count,
            repeat=repeat_result,
        )

    if action is ProviderInvocationRecoveryAction.ESCALATE_HITL:
        binding = validate_provider_invocation_recovery_execution(
            request,
            decision,
            expected_action=ProviderInvocationRecoveryAction.ESCALATE_HITL,
        )
        if not binding.allowed:
            return ProviderInvocationRecoveryExecutionResult(
                decision=decision,
                execution_attempted=True,
                disposition=ProviderInvocationRecoveryExecutionDisposition.BLOCKED,
                provider_mutation_count=0,
                block_reason=binding.block_reason,
            )
        if ports.hitl is None:
            return ProviderInvocationRecoveryExecutionResult(
                decision=decision,
                execution_attempted=True,
                disposition=ProviderInvocationRecoveryExecutionDisposition.BLOCKED,
                provider_mutation_count=0,
                detail="hitl port unavailable",
            )
        escalation = build_recovery_escalation_context(
            request=request,
            decision=decision,
        )
        if escalation is None:
            return ProviderInvocationRecoveryExecutionResult(
                decision=decision,
                execution_attempted=True,
                disposition=ProviderInvocationRecoveryExecutionDisposition.BLOCKED,
                provider_mutation_count=0,
            )
        try:
            hitl_result = ports.hitl.surface_hitl(escalation)
        except Exception as exc:
            return ProviderInvocationRecoveryExecutionResult(
                decision=decision,
                execution_attempted=True,
                disposition=ProviderInvocationRecoveryExecutionDisposition.FAILED,
                provider_mutation_count=0,
                detail=str(exc)[:_MAX_REASON],
            )
        return ProviderInvocationRecoveryExecutionResult(
            decision=decision,
            execution_attempted=True,
            disposition=ProviderInvocationRecoveryExecutionDisposition.COMPLETED,
            provider_mutation_count=0,
            hitl=hitl_result,
        )

    return ProviderInvocationRecoveryExecutionResult(
        decision=decision,
        execution_attempted=False,
        disposition=ProviderInvocationRecoveryExecutionDisposition.BLOCKED,
        provider_mutation_count=0,
        detail="unsupported recovery action",
    )


__all__ = [
    "ProviderInvocationRecoveryExecutionBlockReason",
    "ProviderInvocationRecoveryExecutionDisposition",
    "ProviderInvocationRecoveryExecutionPorts",
    "ProviderInvocationRecoveryExecutionResult",
    "ProviderInvocationRecoveryHitlPort",
    "ProviderInvocationRecoveryHitlResult",
    "ProviderInvocationRecoveryRepeatPort",
    "ProviderInvocationRecoveryRepeatResult",
    "build_recovery_escalation_context",
    "decide_provider_invocation_recovery",
    "execute_provider_invocation_recovery",
]
