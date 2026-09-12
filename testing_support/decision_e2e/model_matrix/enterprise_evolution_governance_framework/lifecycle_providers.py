# © Artur Czarnecki. All rights reserved.

"""Lifecycle completeness providers for evolution governance framework (DS-E2E-15J-L17)."""

from __future__ import annotations

from dataclasses import dataclass

from testing_support.decision_e2e.model_matrix.enterprise_evolution_governance_framework.contracts import (
    EvolutionGovernanceFrameworkContext,
    EvolutionGovernanceIssue,
    EvolutionGovernanceIssueSeverity,
    EvolutionGovernanceLifecycleStage,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.contracts import (
    EvolutionIntelligenceRunStatus,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_strategy.contracts import (
    EvolutionStrategyRunStatus,
)

_DEFAULT_LIFECYCLE_PROVIDER_ID = "default_evolution_lifecycle_governance"
_DEFAULT_LIFECYCLE_PROVIDER_VERSION = "1"


def _issue(
    *,
    issue_id: str,
    severity: EvolutionGovernanceIssueSeverity,
    issue_code: str,
    summary: str,
    stage: EvolutionGovernanceLifecycleStage,
) -> EvolutionGovernanceIssue:
    return EvolutionGovernanceIssue(
        issue_id=issue_id,
        severity=severity,
        issue_code=issue_code,
        summary=summary,
        lifecycle_stage=stage,
        provider_id=_DEFAULT_LIFECYCLE_PROVIDER_ID,
        provider_version=_DEFAULT_LIFECYCLE_PROVIDER_VERSION,
    )


@dataclass(frozen=True, slots=True)
class DefaultEvolutionLifecycleGovernanceProvider:
    @property
    def provider_id(self) -> str:
        return _DEFAULT_LIFECYCLE_PROVIDER_ID

    @property
    def provider_version(self) -> str:
        return _DEFAULT_LIFECYCLE_PROVIDER_VERSION

    def stages_present(
        self,
        context: EvolutionGovernanceFrameworkContext,
    ) -> tuple[EvolutionGovernanceLifecycleStage, ...]:
        stages: list[EvolutionGovernanceLifecycleStage] = []
        if context.intelligence_result is not None:
            stages.append(EvolutionGovernanceLifecycleStage.ANALYSIS)
        if context.strategy_result is not None:
            stages.append(EvolutionGovernanceLifecycleStage.STRATEGY)
        if (
            context.governance_reference is not None
            or context.governance_decision is not None
        ):
            stages.append(EvolutionGovernanceLifecycleStage.GOVERNANCE)
        if context.execution_results:
            stages.append(EvolutionGovernanceLifecycleStage.ADAPTATION)
        if context.operation_records:
            stages.append(EvolutionGovernanceLifecycleStage.OPERATIONS)
        return tuple(stages)

    def assess_lifecycle(
        self,
        context: EvolutionGovernanceFrameworkContext,
    ) -> tuple[EvolutionGovernanceIssue, ...]:
        issues: list[EvolutionGovernanceIssue] = []
        intel = context.intelligence_result
        if intel is None:
            issues.append(
                _issue(
                    issue_id="lifecycle-missing-analysis",
                    severity=EvolutionGovernanceIssueSeverity.INCOMPLETE,
                    issue_code="missing_analysis",
                    summary="Evolution intelligence analysis is not present.",
                    stage=EvolutionGovernanceLifecycleStage.ANALYSIS,
                )
            )
        elif intel.status is not EvolutionIntelligenceRunStatus.COMPLETE:
            issues.append(
                _issue(
                    issue_id="lifecycle-incomplete-analysis",
                    severity=EvolutionGovernanceIssueSeverity.INCOMPLETE,
                    issue_code="incomplete_analysis",
                    summary="Evolution intelligence analysis did not complete.",
                    stage=EvolutionGovernanceLifecycleStage.ANALYSIS,
                )
            )

        strategy = context.strategy_result
        if strategy is None:
            issues.append(
                _issue(
                    issue_id="lifecycle-missing-strategy",
                    severity=EvolutionGovernanceIssueSeverity.INCOMPLETE,
                    issue_code="missing_strategy",
                    summary="Evolution strategy result is not present.",
                    stage=EvolutionGovernanceLifecycleStage.STRATEGY,
                )
            )
        elif strategy.status is not EvolutionStrategyRunStatus.COMPLETE:
            issues.append(
                _issue(
                    issue_id="lifecycle-incomplete-strategy",
                    severity=EvolutionGovernanceIssueSeverity.INCOMPLETE,
                    issue_code="incomplete_strategy",
                    summary="Evolution strategy analysis did not complete.",
                    stage=EvolutionGovernanceLifecycleStage.STRATEGY,
                )
            )

        if context.governance_reference is None and context.governance_decision is None:
            issues.append(
                _issue(
                    issue_id="lifecycle-missing-governance",
                    severity=EvolutionGovernanceIssueSeverity.INCOMPLETE,
                    issue_code="missing_governance_reference",
                    summary="Governance trace (reference or decision) is missing.",
                    stage=EvolutionGovernanceLifecycleStage.GOVERNANCE,
                )
            )

        if not context.execution_results:
            issues.append(
                _issue(
                    issue_id="lifecycle-missing-adaptation",
                    severity=EvolutionGovernanceIssueSeverity.INCOMPLETE,
                    issue_code="missing_adaptation_execution",
                    summary="No adaptation execution results are attached.",
                    stage=EvolutionGovernanceLifecycleStage.ADAPTATION,
                )
            )

        if not context.operation_records:
            issues.append(
                _issue(
                    issue_id="lifecycle-missing-operations",
                    severity=EvolutionGovernanceIssueSeverity.INCOMPLETE,
                    issue_code="missing_operations",
                    summary="No evolution operation records are attached.",
                    stage=EvolutionGovernanceLifecycleStage.OPERATIONS,
                )
            )

        return tuple(issues)


def default_evolution_lifecycle_governance_provider() -> (
    DefaultEvolutionLifecycleGovernanceProvider
):
    return DefaultEvolutionLifecycleGovernanceProvider()


__all__ = [
    "DefaultEvolutionLifecycleGovernanceProvider",
    "default_evolution_lifecycle_governance_provider",
]
