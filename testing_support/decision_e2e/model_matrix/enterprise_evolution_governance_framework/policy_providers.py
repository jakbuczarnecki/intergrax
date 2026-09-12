# © Artur Czarnecki. All rights reserved.

"""Governance policy plugins for evolution governance framework (DS-E2E-15J-L17)."""

from __future__ import annotations

from dataclasses import dataclass

from testing_support.decision_e2e.model_matrix.enterprise_evolution_governance_framework.contracts import (
    EvolutionGovernanceFrameworkContext,
    EvolutionGovernanceIssue,
    EvolutionGovernanceIssueSeverity,
    EvolutionGovernanceLifecycleStage,
)


def _policy_issue(
    *,
    issue_id: str,
    severity: EvolutionGovernanceIssueSeverity,
    issue_code: str,
    summary: str,
    provider_id: str,
    provider_version: str,
    stage: EvolutionGovernanceLifecycleStage | None = None,
) -> EvolutionGovernanceIssue:
    return EvolutionGovernanceIssue(
        issue_id=issue_id,
        severity=severity,
        issue_code=issue_code,
        summary=summary,
        lifecycle_stage=stage,
        provider_id=provider_id,
        provider_version=provider_version,
    )


@dataclass(frozen=True, slots=True)
class EvidenceCompletenessPolicy:
    @property
    def provider_id(self) -> str:
        return "evidence_completeness_policy"

    @property
    def provider_version(self) -> str:
        return "1"

    def assess(
        self,
        context: EvolutionGovernanceFrameworkContext,
    ) -> tuple[EvolutionGovernanceIssue, ...]:
        issues: list[EvolutionGovernanceIssue] = []
        intel = context.intelligence_result
        if intel is not None and not intel.analysis_findings:
            issues.append(
                _policy_issue(
                    issue_id="policy-evidence-intel-findings",
                    severity=EvolutionGovernanceIssueSeverity.REVIEW,
                    issue_code="insufficient_intelligence_evidence",
                    summary="Intelligence analysis completed without recorded findings.",
                    provider_id=self.provider_id,
                    provider_version=self.provider_version,
                    stage=EvolutionGovernanceLifecycleStage.ANALYSIS,
                )
            )
        strategy = context.strategy_result
        if strategy is not None and not strategy.direction_findings:
            issues.append(
                _policy_issue(
                    issue_id="policy-evidence-strategy-findings",
                    severity=EvolutionGovernanceIssueSeverity.REVIEW,
                    issue_code="insufficient_strategy_evidence",
                    summary="Strategy analysis completed without direction findings.",
                    provider_id=self.provider_id,
                    provider_version=self.provider_version,
                    stage=EvolutionGovernanceLifecycleStage.STRATEGY,
                )
            )
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class AuditTraceabilityPolicy:
    @property
    def provider_id(self) -> str:
        return "audit_traceability_policy"

    @property
    def provider_version(self) -> str:
        return "1"

    def assess(
        self,
        context: EvolutionGovernanceFrameworkContext,
    ) -> tuple[EvolutionGovernanceIssue, ...]:
        issues: list[EvolutionGovernanceIssue] = []
        for index, execution in enumerate(context.execution_results):
            if execution.applied_change_reference is None:
                issues.append(
                    _policy_issue(
                        issue_id=f"policy-audit-adaptation-{index}",
                        severity=EvolutionGovernanceIssueSeverity.REVIEW,
                        issue_code="missing_applied_change_reference",
                        summary=(
                            f"Adaptation execution {execution.adaptation_id} "
                            "lacks applied change reference for audit trace."
                        ),
                        provider_id=self.provider_id,
                        provider_version=self.provider_version,
                        stage=EvolutionGovernanceLifecycleStage.ADAPTATION,
                    )
                )
        governance = context.governance_reference
        if governance is not None and not governance.governance_decision_reference:
            issues.append(
                _policy_issue(
                    issue_id="policy-audit-governance-ref",
                    severity=EvolutionGovernanceIssueSeverity.REVIEW,
                    issue_code="missing_governance_decision_reference",
                    summary="Governance reference is missing decision reference id.",
                    provider_id=self.provider_id,
                    provider_version=self.provider_version,
                    stage=EvolutionGovernanceLifecycleStage.GOVERNANCE,
                )
            )
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class LifecycleConsistencyPolicy:
    @property
    def provider_id(self) -> str:
        return "lifecycle_consistency_policy"

    @property
    def provider_version(self) -> str:
        return "1"

    def assess(
        self,
        context: EvolutionGovernanceFrameworkContext,
    ) -> tuple[EvolutionGovernanceIssue, ...]:
        issues: list[EvolutionGovernanceIssue] = []
        if context.intelligence_result is not None and context.strategy_result is None:
            issues.append(
                _policy_issue(
                    issue_id="policy-lifecycle-strategy-after-analysis",
                    severity=EvolutionGovernanceIssueSeverity.REVIEW,
                    issue_code="strategy_missing_after_analysis",
                    summary="Analysis is present but strategy stage is absent.",
                    provider_id=self.provider_id,
                    provider_version=self.provider_version,
                    stage=EvolutionGovernanceLifecycleStage.STRATEGY,
                )
            )
        if context.execution_results and (
            context.governance_reference is None and context.governance_decision is None
        ):
            issues.append(
                _policy_issue(
                    issue_id="policy-lifecycle-governance-before-adaptation",
                    severity=EvolutionGovernanceIssueSeverity.INCOMPLETE,
                    issue_code="adaptation_without_governance_trace",
                    summary="Adaptation results exist without governance trace.",
                    provider_id=self.provider_id,
                    provider_version=self.provider_version,
                    stage=EvolutionGovernanceLifecycleStage.GOVERNANCE,
                )
            )
        return tuple(issues)


def default_evolution_governance_policy_providers() -> tuple[
    EvidenceCompletenessPolicy,
    AuditTraceabilityPolicy,
    LifecycleConsistencyPolicy,
]:
    return (
        EvidenceCompletenessPolicy(),
        AuditTraceabilityPolicy(),
        LifecycleConsistencyPolicy(),
    )


__all__ = [
    "AuditTraceabilityPolicy",
    "EvidenceCompletenessPolicy",
    "LifecycleConsistencyPolicy",
    "default_evolution_governance_policy_providers",
]
