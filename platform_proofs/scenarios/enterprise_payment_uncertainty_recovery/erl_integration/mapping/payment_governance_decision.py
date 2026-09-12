"""Map payment business context to generic platform governance decisions."""

from __future__ import annotations

from decimal import Decimal

from intergrax.contracts.enterprise_reliability.governance_decision import (
    GovernanceDecision,
    GovernanceDisposition,
)
from intergrax.contracts.enterprise_reliability.hitl_requirement import HumanApprovalRequirement
from intergrax.contracts.enterprise_reliability.resolution_decision import ResolutionPlatformAction

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.payment_governance_context import (
    PaymentEnterpriseGovernancePolicy,
    PaymentGovernanceBusinessContext,
)

_HIGH_CUSTOMER_RISK_TIERS = frozenset({"HIGH", "CRITICAL", "ELEVATED"})


def _approval_required(
    *,
    tenant_id: str,
    correlation_id: str,
    contract_id: str,
    requirement_ref: str,
    rationale: str,
) -> GovernanceDecision:
    return GovernanceDecision(
        disposition=GovernanceDisposition.APPROVAL_REQUIRED,
        rationale=rationale,
        hitl_requirement=HumanApprovalRequirement(
            tenant_id=tenant_id,
            correlation_id=correlation_id,
            contract_id=contract_id,
            requirement_ref=requirement_ref,
            rationale=rationale,
        ),
    )


def decide_payment_governance(
    *,
    business_context: PaymentGovernanceBusinessContext,
    policy: PaymentEnterpriseGovernancePolicy,
    resolution_action: ResolutionPlatformAction,
    tenant_id: str,
    contract_id: str,
    correlation_id: str,
) -> GovernanceDecision:
    """
    Enterprise payment policy — returns only platform ``GovernanceDecision`` values.
    """
    if resolution_action is not ResolutionPlatformAction.CONTINUE:
        return GovernanceDecision(
            disposition=GovernanceDisposition.ALLOW,
            rationale="no_risky_continuation_proposed",
        )

    amount = business_context.payment_amount
    if amount is None:
        return _approval_required(
            tenant_id=tenant_id,
            correlation_id=correlation_id,
            contract_id=contract_id,
            requirement_ref="erl-qual-004:governance:insufficient_payment_context",
            rationale="payment_amount_unavailable_for_governance",
        )

    if business_context.currency.strip().upper() != policy.currency.strip().upper():
        return _approval_required(
            tenant_id=tenant_id,
            correlation_id=correlation_id,
            contract_id=contract_id,
            requirement_ref="erl-qual-004:governance:currency_mismatch",
            rationale="payment_currency_not_covered_by_policy",
        )

    risk_tier = (business_context.customer_risk_tier or "").strip().upper()
    if risk_tier in _HIGH_CUSTOMER_RISK_TIERS:
        return _approval_required(
            tenant_id=tenant_id,
            correlation_id=correlation_id,
            contract_id=contract_id,
            requirement_ref="erl-qual-004:governance:elevated_customer_risk",
            rationale="customer_risk_requires_human_approval",
        )

    if amount >= policy.human_approval_threshold_amount:
        return _approval_required(
            tenant_id=tenant_id,
            correlation_id=correlation_id,
            contract_id=contract_id,
            requirement_ref="erl-qual-004:governance:high_value_payment",
            rationale="payment_amount_exceeds_auto_allow_threshold",
        )

    if amount < Decimal("0"):
        return GovernanceDecision(
            disposition=GovernanceDisposition.DENY,
            rationale="invalid_negative_payment_amount",
        )

    return GovernanceDecision(
        disposition=GovernanceDisposition.ALLOW,
        rationale="payment_within_auto_allow_policy",
    )
