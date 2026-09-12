# © Artur Czarnecki. All rights reserved.

"""Controlled enterprise adaptation orchestration via injected plugins (DS-E2E-15J-L13)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from testing_support.decision_e2e.model_matrix.autonomous_enterprise_adaptation.contracts import (
    AdaptationExecutionResult,
    AdaptationExecutionStatus,
    ApprovedAdaptationRequest,
)
from testing_support.decision_e2e.model_matrix.autonomous_enterprise_adaptation.protocol import (
    AdaptationAuditProvider,
    AdaptationApplyOutcome,
    EnterpriseAdaptationProvider,
)
from testing_support.decision_e2e.model_matrix.self_improvement_governance.contracts import (
    SelfImprovementGovernanceStatus,
)


def _governance_gate(
    request: ApprovedAdaptationRequest,
) -> AdaptationApplyOutcome | None:
    approval = request.governance_approval
    if approval is None:
        return AdaptationApplyOutcome(
            status=AdaptationExecutionStatus.REJECTED,
            applied_change_reference=None,
            summary="Adaptation rejected: missing governance approval reference.",
        )
    if approval.governance_status is not SelfImprovementGovernanceStatus.APPROVED:
        return AdaptationApplyOutcome(
            status=AdaptationExecutionStatus.REJECTED,
            applied_change_reference=None,
            summary=(
                f"Adaptation rejected: governance status "
                f"{approval.governance_status.value} is not approved."
            ),
        )
    return None


def _scope_within_constraints(
    request: ApprovedAdaptationRequest,
    *,
    evaluated_at: datetime,
) -> AdaptationApplyOutcome | None:
    scope = request.scope
    if not request.constraints:
        return AdaptationApplyOutcome(
            status=AdaptationExecutionStatus.REJECTED,
            applied_change_reference=None,
            summary="Adaptation rejected: no constraints defined for controlled scope.",
        )
    matching = [
        item
        for item in request.constraints
        if item.scope_id == scope.scope_id
        and item.system_area == scope.target_system_area
    ]
    if not matching:
        return AdaptationApplyOutcome(
            status=AdaptationExecutionStatus.REJECTED,
            applied_change_reference=None,
            summary=(
                f"Adaptation rejected: scope {scope.scope_id} "
                f"outside allowed constraint boundaries."
            ),
        )
    for constraint in matching:
        if constraint.valid_until is not None and evaluated_at > constraint.valid_until:
            return AdaptationApplyOutcome(
                status=AdaptationExecutionStatus.REJECTED,
                applied_change_reference=None,
                summary=(
                    f"Adaptation rejected: constraint {constraint.constraint_id} "
                    "is no longer valid."
                ),
            )
        if constraint.adaptation_version != request.version:
            return AdaptationApplyOutcome(
                status=AdaptationExecutionStatus.REJECTED,
                applied_change_reference=None,
                summary=(
                    f"Adaptation rejected: version {request.version} "
                    f"not permitted by constraint {constraint.constraint_id}."
                ),
            )
    return None


def _finalize(
    request: ApprovedAdaptationRequest,
    *,
    outcome: AdaptationApplyOutcome,
    adaptation_provider: EnterpriseAdaptationProvider,
    audit_provider: AdaptationAuditProvider,
    executed_at: datetime,
) -> AdaptationExecutionResult:
    audit = audit_provider.build_audit(
        request,
        outcome_status=outcome.status,
        adaptation_provider_id=adaptation_provider.provider_id,
        adaptation_provider_version=adaptation_provider.provider_version,
        applied_change_reference=outcome.applied_change_reference,
        executed_at=executed_at,
        outcome_summary=outcome.summary,
    )
    return AdaptationExecutionResult(
        status=outcome.status,
        provider_id=adaptation_provider.provider_id,
        provider_version=adaptation_provider.provider_version,
        applied_change_reference=outcome.applied_change_reference,
        audit_metadata=audit,
        adaptation_id=request.adaptation_id,
        version=request.version,
        source_reference=request.source_reference,
    )


@dataclass(frozen=True, slots=True)
class AutonomousEnterpriseAdaptationEngine:
    adaptation_provider: EnterpriseAdaptationProvider
    audit_provider: AdaptationAuditProvider

    def apply_adaptation(
        self,
        approved_change: ApprovedAdaptationRequest,
        *,
        executed_at: datetime | None = None,
    ) -> AdaptationExecutionResult:
        stamp = executed_at or datetime.now(tz=UTC)
        gate = _governance_gate(approved_change)
        if gate is not None:
            return _finalize(
                approved_change,
                outcome=gate,
                adaptation_provider=self.adaptation_provider,
                audit_provider=self.audit_provider,
                executed_at=stamp,
            )
        scope_gate = _scope_within_constraints(approved_change, evaluated_at=stamp)
        if scope_gate is not None:
            return _finalize(
                approved_change,
                outcome=scope_gate,
                adaptation_provider=self.adaptation_provider,
                audit_provider=self.audit_provider,
                executed_at=stamp,
            )
        provider_outcome = self.adaptation_provider.apply_adaptation(approved_change)
        return _finalize(
            approved_change,
            outcome=provider_outcome,
            adaptation_provider=self.adaptation_provider,
            audit_provider=self.audit_provider,
            executed_at=stamp,
        )


def default_autonomous_enterprise_adaptation_engine(
    *,
    adaptation_provider: EnterpriseAdaptationProvider | None = None,
    audit_provider: AdaptationAuditProvider | None = None,
) -> AutonomousEnterpriseAdaptationEngine:
    if audit_provider is None:
        from testing_support.decision_e2e.model_matrix.autonomous_enterprise_adaptation.audit_providers import (
            default_adaptation_audit_provider,
        )

        audit_provider = default_adaptation_audit_provider()
    if adaptation_provider is None:
        from testing_support.decision_e2e.model_matrix.autonomous_enterprise_adaptation.adaptation_providers import (
            DefaultEnterpriseAdaptationProvider,
        )

        adaptation_provider = DefaultEnterpriseAdaptationProvider()
    return AutonomousEnterpriseAdaptationEngine(
        adaptation_provider=adaptation_provider,
        audit_provider=audit_provider,
    )


__all__ = [
    "AutonomousEnterpriseAdaptationEngine",
    "default_autonomous_enterprise_adaptation_engine",
]
