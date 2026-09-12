"""In-memory payment recovery action port — records executions for unit tests."""

from __future__ import annotations

from dataclasses import dataclass, field

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.payment_recovery_action import (
    PaymentRecoveryActionResult,
    PaymentRecoveryBusinessAction,
    PaymentRecoveryExecutionStatus,
)


@dataclass
class InMemoryPaymentRecoveryActionPort:
    """Replaceable stub — no singleton; inject per test or wiring bundle."""

    _executions: list[PaymentRecoveryActionResult] = field(default_factory=list)
    _last_correlation_id: str | None = field(default=None, init=False)

    @property
    def executions(self) -> tuple[PaymentRecoveryActionResult, ...]:
        return tuple(self._executions)

    @property
    def last_correlation_id(self) -> str | None:
        return self._last_correlation_id

    def execute(
        self,
        *,
        correlation_id: str,
        tenant_id: str,
        action: PaymentRecoveryBusinessAction,
        resolution_rationale: str,
    ) -> PaymentRecoveryActionResult:
        _ = tenant_id
        _ = resolution_rationale
        self._last_correlation_id = correlation_id
        if action is PaymentRecoveryBusinessAction.RESUME_FULFILLMENT:
            result = PaymentRecoveryActionResult(
                action=action,
                status=PaymentRecoveryExecutionStatus.SUCCESS,
                detail="fulfillment_workflow_resumed",
            )
        elif action is PaymentRecoveryBusinessAction.RELEASE_RESERVATION:
            result = PaymentRecoveryActionResult(
                action=action,
                status=PaymentRecoveryExecutionStatus.SUCCESS,
                detail="reservation_released_business_flow_stopped",
            )
        else:
            status = (
                PaymentRecoveryExecutionStatus.WAITING
                if "wait" in resolution_rationale.lower()
                else PaymentRecoveryExecutionStatus.ESCALATED
            )
            result = PaymentRecoveryActionResult(
                action=action,
                status=status,
                detail="operational_follow_up_recorded",
            )
        self._executions.append(result)
        return result
