# © Artur Czarnecki. All rights reserved.

"""Pluggable evolution compliance validators (DS-E2E-15J-L18)."""

from __future__ import annotations

from dataclasses import dataclass

from testing_support.decision_e2e.model_matrix.enterprise_evolution_assurance.contracts import (
    EvolutionAssuranceContext,
    EvolutionAssuranceFinding,
    EvolutionAssuranceFindingSeverity,
    EvolutionAssuranceLifecycleStage,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_governance_framework.contracts import (
    EvolutionGovernanceFrameworkStatus,
)

_REQUIRED_STAGES_ID = "required_stages_compliance_validator"
_REQUIRED_STAGES_VERSION = "1"
_FRAMEWORK_ALIGNMENT_ID = "governance_framework_alignment_validator"
_FRAMEWORK_ALIGNMENT_VERSION = "1"


@dataclass(frozen=True, slots=True)
class RequiredStagesComplianceValidator:
    @property
    def provider_id(self) -> str:
        return _REQUIRED_STAGES_ID

    @property
    def provider_version(self) -> str:
        return _REQUIRED_STAGES_VERSION

    def validate(
        self,
        context: EvolutionAssuranceContext,
    ) -> tuple[EvolutionAssuranceFinding, ...]:
        findings: list[EvolutionAssuranceFinding] = []
        if context.intelligence_result is None:
            findings.append(
                EvolutionAssuranceFinding(
                    finding_id="compliance-missing-intelligence",
                    severity=EvolutionAssuranceFindingSeverity.FAILED,
                    finding_code="required_stage_intelligence",
                    summary="Intelligence stage evidence is required for compliance.",
                    lifecycle_stage=EvolutionAssuranceLifecycleStage.ANALYSIS,
                    provider_id=self.provider_id,
                    provider_version=self.provider_version,
                )
            )
        if context.strategy_result is None:
            findings.append(
                EvolutionAssuranceFinding(
                    finding_id="compliance-missing-strategy",
                    severity=EvolutionAssuranceFindingSeverity.FAILED,
                    finding_code="required_stage_strategy",
                    summary="Strategy stage evidence is required for compliance.",
                    lifecycle_stage=EvolutionAssuranceLifecycleStage.STRATEGY,
                    provider_id=self.provider_id,
                    provider_version=self.provider_version,
                )
            )
        if context.governance_reference is None and context.governance_decision is None:
            findings.append(
                EvolutionAssuranceFinding(
                    finding_id="compliance-missing-governance",
                    severity=EvolutionAssuranceFindingSeverity.REVIEW,
                    finding_code="required_stage_governance",
                    summary="Governance stage reference or decision is required.",
                    lifecycle_stage=EvolutionAssuranceLifecycleStage.GOVERNANCE,
                    provider_id=self.provider_id,
                    provider_version=self.provider_version,
                )
            )
        return tuple(findings)


@dataclass(frozen=True, slots=True)
class GovernanceFrameworkAlignmentValidator:
    @property
    def provider_id(self) -> str:
        return _FRAMEWORK_ALIGNMENT_ID

    @property
    def provider_version(self) -> str:
        return _FRAMEWORK_ALIGNMENT_VERSION

    def validate(
        self,
        context: EvolutionAssuranceContext,
    ) -> tuple[EvolutionAssuranceFinding, ...]:
        findings: list[EvolutionAssuranceFinding] = []
        framework = context.governance_framework_result
        if framework is None:
            findings.append(
                EvolutionAssuranceFinding(
                    finding_id="compliance-missing-framework-result",
                    severity=EvolutionAssuranceFindingSeverity.REVIEW,
                    finding_code="missing_governance_framework_result",
                    summary="Governance framework evaluation result is not attached.",
                    lifecycle_stage=EvolutionAssuranceLifecycleStage.GOVERNANCE_FRAMEWORK,
                    provider_id=self.provider_id,
                    provider_version=self.provider_version,
                )
            )
            return tuple(findings)
        if framework.status is EvolutionGovernanceFrameworkStatus.BLOCKED:
            findings.append(
                EvolutionAssuranceFinding(
                    finding_id="compliance-framework-blocked",
                    severity=EvolutionAssuranceFindingSeverity.FAILED,
                    finding_code="governance_framework_blocked",
                    summary="Governance framework reported blocked status.",
                    lifecycle_stage=EvolutionAssuranceLifecycleStage.GOVERNANCE_FRAMEWORK,
                    provider_id=self.provider_id,
                    provider_version=self.provider_version,
                    evidence_ref=framework.framework_task_id,
                )
            )
        if context.scope_id and framework.audit.scope_id != context.scope_id:
            findings.append(
                EvolutionAssuranceFinding(
                    finding_id="compliance-framework-scope-mismatch",
                    severity=EvolutionAssuranceFindingSeverity.REVIEW,
                    finding_code="governance_framework_scope_mismatch",
                    summary=(
                        "Governance framework scope_id does not match assurance context."
                    ),
                    lifecycle_stage=EvolutionAssuranceLifecycleStage.GOVERNANCE_FRAMEWORK,
                    provider_id=self.provider_id,
                    provider_version=self.provider_version,
                )
            )
        return tuple(findings)


def default_evolution_compliance_validator_providers() -> tuple[
    RequiredStagesComplianceValidator,
    GovernanceFrameworkAlignmentValidator,
]:
    return (
        RequiredStagesComplianceValidator(),
        GovernanceFrameworkAlignmentValidator(),
    )


__all__ = [
    "GovernanceFrameworkAlignmentValidator",
    "RequiredStagesComplianceValidator",
    "default_evolution_compliance_validator_providers",
]
