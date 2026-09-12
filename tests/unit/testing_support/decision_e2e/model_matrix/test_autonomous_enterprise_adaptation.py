# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from testing_support.decision_e2e.model_matrix.autonomous_enterprise_adaptation import (
    AdaptationApplyOutcome,
    AdaptationConstraint,
    AdaptationExecutionStatus,
    AdaptationScope,
    ApprovedAdaptationRequest,
    AutonomousEnterpriseAdaptationEngine,
    EvolutionSourceReference,
    GovernanceApprovalReference,
    default_adaptation_audit_provider,
    default_autonomous_enterprise_adaptation_engine,
)
from testing_support.decision_e2e.model_matrix.self_improvement_governance.contracts import (
    SelfImprovementGovernanceStatus,
)


def _stamp() -> datetime:
    return datetime(2026, 9, 12, 18, 0, 0, tzinfo=UTC)


def _source() -> EvolutionSourceReference:
    return EvolutionSourceReference(
        controlled_evolution_record_id="evo-rec-1",
        proposal_id="prop-routing-1",
        source_insight_ids=("insight-1",),
    )


def _approval(
    *,
    status: SelfImprovementGovernanceStatus = SelfImprovementGovernanceStatus.APPROVED,
) -> GovernanceApprovalReference:
    return GovernanceApprovalReference(
        approval_id="gov-appr-1",
        governance_status=status,
        approver_identity="enterprise-governance-board",
        governance_decision_reference="gov-decision-1",
    )


def _scope(*, scope_id: str = "scope-routing-shadow") -> AdaptationScope:
    return AdaptationScope(
        scope_id=scope_id,
        target_system_area="model_matrix.routing",
        scope_label="Shadow model routing for document queries",
    )


def _constraints(
    *, scope_id: str = "scope-routing-shadow"
) -> tuple[AdaptationConstraint, ...]:
    return (
        AdaptationConstraint(
            constraint_id="cstr-1",
            scope_id=scope_id,
            system_area="model_matrix.routing",
            adaptation_version="1",
            valid_until=datetime(2027, 1, 1, tzinfo=UTC),
        ),
    )


def _request(
    *,
    governance: GovernanceApprovalReference | None = _approval(),
    scope: AdaptationScope | None = None,
    constraints: tuple[AdaptationConstraint, ...] | None = None,
) -> ApprovedAdaptationRequest:
    resolved_scope = scope or _scope()
    return ApprovedAdaptationRequest(
        adaptation_id="adapt-1",
        version="1",
        source_reference=_source(),
        governance_approval=governance,
        scope=resolved_scope,
        constraints=constraints if constraints is not None else _constraints(),
    )


def test_default_adaptation_provider_applies_governed_change() -> None:
    engine = default_autonomous_enterprise_adaptation_engine()
    result = engine.apply_adaptation(_request(), executed_at=_stamp())
    assert result.status is AdaptationExecutionStatus.APPLIED
    assert result.provider_id == "default_enterprise_adaptation"
    assert result.applied_change_reference == "adapted:adapt-1:v1:prop-routing-1"


@dataclass(frozen=True, slots=True)
class CustomEnterpriseAdaptationProvider:
    @property
    def provider_id(self) -> str:
        return "custom_enterprise_adaptation"

    @property
    def provider_version(self) -> str:
        return "9"

    def apply_adaptation(
        self, approved_change: ApprovedAdaptationRequest
    ) -> AdaptationApplyOutcome:
        return AdaptationApplyOutcome(
            status=AdaptationExecutionStatus.APPLIED,
            applied_change_reference=f"custom:{approved_change.adaptation_id}",
            summary="Custom provider applied governed adaptation.",
        )


def test_adaptation_provider_swap_without_engine_change() -> None:
    engine = AutonomousEnterpriseAdaptationEngine(
        adaptation_provider=CustomEnterpriseAdaptationProvider(),
        audit_provider=default_adaptation_audit_provider(),
    )
    result = engine.apply_adaptation(_request(), executed_at=_stamp())
    assert result.status is AdaptationExecutionStatus.APPLIED
    assert result.provider_id == "custom_enterprise_adaptation"
    assert result.provider_version == "9"
    assert result.applied_change_reference == "custom:adapt-1"


def test_adaptation_rejected_without_governance_approval() -> None:
    engine = default_autonomous_enterprise_adaptation_engine()
    result = engine.apply_adaptation(
        _request(governance=None),
        executed_at=_stamp(),
    )
    assert result.status is AdaptationExecutionStatus.REJECTED
    assert result.applied_change_reference is None


def test_adaptation_audit_records_provider_version_and_source() -> None:
    engine = default_autonomous_enterprise_adaptation_engine()
    result = engine.apply_adaptation(_request(), executed_at=_stamp())
    audit = result.audit_metadata
    assert audit.provider_id == "default_enterprise_adaptation"
    assert audit.provider_version == "1"
    assert audit.source_reference.controlled_evolution_record_id == "evo-rec-1"
    assert audit.source_reference.proposal_id == "prop-routing-1"
    assert audit.governance_approval_id == "gov-appr-1"
    assert audit.approver_identity == "enterprise-governance-board"


def test_adaptation_rejected_when_scope_outside_constraints() -> None:
    engine = default_autonomous_enterprise_adaptation_engine()
    out_of_scope = _scope(scope_id="scope-production-global")
    result = engine.apply_adaptation(
        _request(
            scope=out_of_scope,
            constraints=_constraints(scope_id="scope-routing-shadow"),
        ),
        executed_at=_stamp(),
    )
    assert result.status is AdaptationExecutionStatus.REJECTED
    assert "outside allowed constraint" in result.audit_metadata.outcome_summary
