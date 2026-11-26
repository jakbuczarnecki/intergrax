# © Artur Czarnecki. All rights reserved.

"""ARCH-FREEZE package F — UNKNOWN recovery must not auto-mutate providers."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from intergrax.contracts.enterprise_reliability.compensation import (
    CompensationDisposition,
    CompensationPlan,
)
from intergrax.contracts.enterprise_reliability.compensation_decision import (
    CompensationDecision,
    CompensationPlatformIntent,
)
from intergrax.contracts.enterprise_reliability.compensation_execution import (
    CompensationExecutionOutcome,
)
from intergrax.contracts.enterprise_reliability.effect_contract import (
    ExternalEffectCapabilitySupport,
    ExternalEffectCategory,
    ExternalEffectContract,
    ExternalEffectOutcome,
    ExternalEffectSafetyCapabilities,
    evaluate_unknown_uncertainty_posture,
)
from intergrax.contracts.enterprise_reliability.provider_invocation_recovery import (
    ProviderInvocationRecoveryAction,
    ProviderInvocationRecoveryDispatchState,
    ProviderInvocationRecoveryPolicyRequest,
)
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
    ProviderInvocationStatus,
)
from intergrax.runtime.enterprise_reliability.compensation_execution import (
    execute_external_effect_compensation,
)
from intergrax.runtime.enterprise_reliability.compensation_orchestration import (
    ExternalEffectCompensationPlanning,
)
from intergrax.runtime.enterprise_reliability.default_provider_invocation_recovery_policy import (
    DefaultFailClosedProviderInvocationRecoveryPolicy,
)
from intergrax.runtime.enterprise_reliability.uncertainty_lifecycle import (
    admit_external_effect_unknown,
)

pytestmark = [pytest.mark.unit]


def _invocation() -> ProviderInvocation:
    return ProviderInvocation(
        invocation_id="inv-af-f",
        idempotency_key="idem-af-f",
        operation="external_work.capture_payment",
        provider_id="provider.test",
        created_at=datetime.now(tz=UTC),
    )


def _contract() -> ExternalEffectContract:
    return ExternalEffectContract(
        contract_id="contract.af.f",
        operation_key="external_work.capture_payment",
        category=ExternalEffectCategory.FINANCIAL,
        safety=ExternalEffectSafetyCapabilities(
            idempotency=ExternalEffectCapabilitySupport.SUPPORTED,
            reconciliation=ExternalEffectCapabilitySupport.SUPPORTED,
            compensation=ExternalEffectCapabilitySupport.SUPPORTED,
        ),
    )


def test_default_policy_unknown_does_not_pick_mutating_allowed_first() -> None:
    policy = DefaultFailClosedProviderInvocationRecoveryPolicy()
    inv = _invocation()
    outcome = ProviderInvocationOutcome(
        invocation_id=inv.invocation_id,
        status=ProviderInvocationStatus.UNKNOWN,
        recorded_at=datetime.now(tz=UTC),
    )
    request = ProviderInvocationRecoveryPolicyRequest(
        invocation=inv,
        outcome=outcome,
        dispatch_state=ProviderInvocationRecoveryDispatchState.OUTCOME_RECORDED,
        effect_contract=_contract(),
        unknown_posture=evaluate_unknown_uncertainty_posture(_contract()),
        reconciliation=None,
        repeat_eligibility=None,
        allowed_actions=(
            ProviderInvocationRecoveryAction.IDEMPOTENT_REPEAT,
            ProviderInvocationRecoveryAction.ESCALATE_HITL,
        ),
    )
    decision = policy.decide(request)
    assert decision.selected_action is ProviderInvocationRecoveryAction.ESCALATE_HITL


def test_unknown_compensation_plan_escalate_does_not_invoke_plugin() -> None:
    state = admit_external_effect_unknown(correlation_id="corr-af-f")
    assert state.effect_outcome is ExternalEffectOutcome.UNKNOWN
    planning = ExternalEffectCompensationPlanning(
        state=state,
        contract_id=_contract().contract_id,
        plan=CompensationPlan(
            disposition=CompensationDisposition.ESCALATE_REQUIRED,
            decision=CompensationDecision(
                intent=CompensationPlatformIntent.ESCALATE,
                rationale="insufficient definitive evidence",
            ),
            rationale="insufficient definitive evidence",
        ),
    )
    gateway = object()
    run = execute_external_effect_compensation(
        planning=planning,
        effect_contract=_contract(),
        gateway=gateway,  # type: ignore[arg-type]
        tenant_id="test-tenant",
    )
    assert run.execution.outcome is CompensationExecutionOutcome.ESCALATED
    assert run.execution.plugin_result is None
