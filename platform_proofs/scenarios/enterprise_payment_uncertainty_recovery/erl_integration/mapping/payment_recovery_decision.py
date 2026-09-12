"""Payment recovery rules — platform ``RecoveryDecision`` plus scenario business execution."""

from __future__ import annotations

from intergrax.contracts.enterprise_reliability.recovery_decision import (
    RecoveryDecision,
    RecoveryLifecycleAction,
)
from intergrax.contracts.enterprise_reliability.resolution_decision import (
    ResolutionDecision,
    ResolutionPlatformAction,
)

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.payment_recovery_action import (
    PaymentRecoveryActionPort,
    PaymentRecoveryActionResult,
    PaymentRecoveryBusinessAction,
    PaymentRecoveryExecutionStatus,
)


def decide_payment_recovery(
    *,
    resolution_decision: ResolutionDecision,
    action_port: PaymentRecoveryActionPort,
    tenant_id: str,
    correlation_id: str,
) -> tuple[RecoveryDecision, PaymentRecoveryActionResult]:
    """
    Map platform resolution to lifecycle recommendation and execute payment recovery.

    Driven by ``resolution_decision.action`` — never by scenario variant identifiers.
    """
    action = resolution_decision.action
    rationale = resolution_decision.rationale

    if action is ResolutionPlatformAction.CONTINUE:
        business_action = PaymentRecoveryBusinessAction.RESUME_FULFILLMENT
        lifecycle = RecoveryLifecycleAction.CONTINUE
        recovery_rationale = "payment_confirmed_resume_fulfillment"
    elif action is ResolutionPlatformAction.STOP:
        business_action = PaymentRecoveryBusinessAction.RELEASE_RESERVATION
        lifecycle = RecoveryLifecycleAction.TERMINATE
        recovery_rationale = "payment_failed_release_reservation_stop_flow"
    elif action is ResolutionPlatformAction.UNKNOWN:
        business_action = PaymentRecoveryBusinessAction.OPERATIONAL_FOLLOW_UP
        lifecycle = RecoveryLifecycleAction.WAIT
        recovery_rationale = "payment_truth_unresolved_operational_wait"
    elif action in {
        ResolutionPlatformAction.ESCALATE,
        ResolutionPlatformAction.COMPENSATION_REQUIRED,
    }:
        business_action = PaymentRecoveryBusinessAction.OPERATIONAL_FOLLOW_UP
        lifecycle = RecoveryLifecycleAction.ESCALATE
        recovery_rationale = "payment_uncertainty_escalate_operations"
    else:
        business_action = PaymentRecoveryBusinessAction.OPERATIONAL_FOLLOW_UP
        lifecycle = RecoveryLifecycleAction.ESCALATE
        recovery_rationale = f"payment_recovery_unmapped_resolution:{action.value}"

    execution = action_port.execute(
        correlation_id=correlation_id,
        tenant_id=tenant_id,
        action=business_action,
        resolution_rationale=(
            recovery_rationale
            if business_action is PaymentRecoveryBusinessAction.OPERATIONAL_FOLLOW_UP
            else (rationale or recovery_rationale)
        ),
    )
    if business_action is PaymentRecoveryBusinessAction.OPERATIONAL_FOLLOW_UP:
        if execution.status is PaymentRecoveryExecutionStatus.WAITING:
            lifecycle = RecoveryLifecycleAction.WAIT
        elif execution.status is PaymentRecoveryExecutionStatus.ESCALATED:
            lifecycle = RecoveryLifecycleAction.ESCALATE

    return (
        RecoveryDecision(action=lifecycle, rationale=recovery_rationale),
        execution,
    )
