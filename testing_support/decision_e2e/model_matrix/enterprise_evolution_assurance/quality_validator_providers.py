# © Artur Czarnecki. All rights reserved.

"""Pluggable evolution quality validators (DS-E2E-15J-L18)."""

from __future__ import annotations

from dataclasses import dataclass

from testing_support.decision_e2e.model_matrix.enterprise_evolution_assurance.contracts import (
    EvolutionAssuranceContext,
    EvolutionAssuranceFinding,
    EvolutionAssuranceFindingSeverity,
    EvolutionAssuranceLifecycleStage,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.contracts import (
    EvolutionIntelligenceRunStatus,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_strategy.contracts import (
    EvolutionStrategyRunStatus,
)

_LIFECYCLE_COMPLETENESS_ID = "lifecycle_completeness_validator"
_LIFECYCLE_COMPLETENESS_VERSION = "1"
_EVIDENCE_AVAILABILITY_ID = "evidence_availability_validator"
_EVIDENCE_AVAILABILITY_VERSION = "1"
_VERSION_CONSISTENCY_ID = "version_consistency_validator"
_VERSION_CONSISTENCY_VERSION = "1"


def _finding(
    *,
    finding_id: str,
    severity: EvolutionAssuranceFindingSeverity,
    finding_code: str,
    summary: str,
    stage: EvolutionAssuranceLifecycleStage | None,
    provider_id: str,
    provider_version: str,
    evidence_ref: str | None = None,
) -> EvolutionAssuranceFinding:
    return EvolutionAssuranceFinding(
        finding_id=finding_id,
        severity=severity,
        finding_code=finding_code,
        summary=summary,
        lifecycle_stage=stage,
        provider_id=provider_id,
        provider_version=provider_version,
        evidence_ref=evidence_ref,
    )


@dataclass(frozen=True, slots=True)
class LifecycleCompletenessValidator:
    @property
    def provider_id(self) -> str:
        return _LIFECYCLE_COMPLETENESS_ID

    @property
    def provider_version(self) -> str:
        return _LIFECYCLE_COMPLETENESS_VERSION

    def validate(
        self,
        context: EvolutionAssuranceContext,
    ) -> tuple[EvolutionAssuranceFinding, ...]:
        findings: list[EvolutionAssuranceFinding] = []
        intel = context.intelligence_result
        if intel is None:
            findings.append(
                _finding(
                    finding_id="quality-missing-analysis",
                    severity=EvolutionAssuranceFindingSeverity.REVIEW,
                    finding_code="missing_analysis",
                    summary="Evolution intelligence result is absent from assurance context.",
                    stage=EvolutionAssuranceLifecycleStage.ANALYSIS,
                    provider_id=self.provider_id,
                    provider_version=self.provider_version,
                )
            )
        elif intel.status is not EvolutionIntelligenceRunStatus.COMPLETE:
            findings.append(
                _finding(
                    finding_id="quality-incomplete-analysis",
                    severity=EvolutionAssuranceFindingSeverity.REVIEW,
                    finding_code="incomplete_analysis",
                    summary="Evolution intelligence analysis did not complete successfully.",
                    stage=EvolutionAssuranceLifecycleStage.ANALYSIS,
                    provider_id=self.provider_id,
                    provider_version=self.provider_version,
                )
            )
        strategy = context.strategy_result
        if strategy is None:
            findings.append(
                _finding(
                    finding_id="quality-missing-strategy",
                    severity=EvolutionAssuranceFindingSeverity.REVIEW,
                    finding_code="missing_strategy",
                    summary="Evolution strategy result is absent from assurance context.",
                    stage=EvolutionAssuranceLifecycleStage.STRATEGY,
                    provider_id=self.provider_id,
                    provider_version=self.provider_version,
                )
            )
        elif strategy.status is not EvolutionStrategyRunStatus.COMPLETE:
            findings.append(
                _finding(
                    finding_id="quality-incomplete-strategy",
                    severity=EvolutionAssuranceFindingSeverity.REVIEW,
                    finding_code="incomplete_strategy",
                    summary="Evolution strategy analysis did not complete successfully.",
                    stage=EvolutionAssuranceLifecycleStage.STRATEGY,
                    provider_id=self.provider_id,
                    provider_version=self.provider_version,
                )
            )
        if not context.execution_results:
            findings.append(
                _finding(
                    finding_id="quality-missing-adaptation",
                    severity=EvolutionAssuranceFindingSeverity.WARNING,
                    finding_code="missing_adaptation",
                    summary="No adaptation execution results are present.",
                    stage=EvolutionAssuranceLifecycleStage.ADAPTATION,
                    provider_id=self.provider_id,
                    provider_version=self.provider_version,
                )
            )
        if not context.operation_records:
            findings.append(
                _finding(
                    finding_id="quality-missing-operations",
                    severity=EvolutionAssuranceFindingSeverity.WARNING,
                    finding_code="missing_operations",
                    summary="No evolution operation records are present.",
                    stage=EvolutionAssuranceLifecycleStage.OPERATIONS,
                    provider_id=self.provider_id,
                    provider_version=self.provider_version,
                )
            )
        return tuple(findings)


@dataclass(frozen=True, slots=True)
class EvidenceAvailabilityValidator:
    @property
    def provider_id(self) -> str:
        return _EVIDENCE_AVAILABILITY_ID

    @property
    def provider_version(self) -> str:
        return _EVIDENCE_AVAILABILITY_VERSION

    def validate(
        self,
        context: EvolutionAssuranceContext,
    ) -> tuple[EvolutionAssuranceFinding, ...]:
        findings: list[EvolutionAssuranceFinding] = []
        if context.process_reference is None:
            findings.append(
                _finding(
                    finding_id="quality-missing-process-ref",
                    severity=EvolutionAssuranceFindingSeverity.REVIEW,
                    finding_code="missing_process_reference",
                    summary="Process reference is not attached to assurance context.",
                    stage=None,
                    provider_id=self.provider_id,
                    provider_version=self.provider_version,
                )
            )
        if context.governance_reference is None and context.governance_decision is None:
            findings.append(
                _finding(
                    finding_id="quality-missing-governance-evidence",
                    severity=EvolutionAssuranceFindingSeverity.REVIEW,
                    finding_code="missing_governance_evidence",
                    summary="Governance decision or reference evidence is missing.",
                    stage=EvolutionAssuranceLifecycleStage.GOVERNANCE,
                    provider_id=self.provider_id,
                    provider_version=self.provider_version,
                )
            )
        return tuple(findings)


@dataclass(frozen=True, slots=True)
class VersionConsistencyValidator:
    @property
    def provider_id(self) -> str:
        return _VERSION_CONSISTENCY_ID

    @property
    def provider_version(self) -> str:
        return _VERSION_CONSISTENCY_VERSION

    def validate(
        self,
        context: EvolutionAssuranceContext,
    ) -> tuple[EvolutionAssuranceFinding, ...]:
        findings: list[EvolutionAssuranceFinding] = []
        process = context.process_reference
        if process is not None and process.version != context.version:
            findings.append(
                _finding(
                    finding_id="quality-process-version-mismatch",
                    severity=EvolutionAssuranceFindingSeverity.WARNING,
                    finding_code="process_version_mismatch",
                    summary=(
                        f"Process reference version {process.version} "
                        f"differs from scope version {context.version}."
                    ),
                    stage=None,
                    provider_id=self.provider_id,
                    provider_version=self.provider_version,
                    evidence_ref=f"process:{process.process_id}",
                )
            )
        for index, execution in enumerate(context.execution_results):
            if execution.version != context.version and context.version:
                findings.append(
                    _finding(
                        finding_id=f"quality-adaptation-version-{index}",
                        severity=EvolutionAssuranceFindingSeverity.WARNING,
                        finding_code="adaptation_version_mismatch",
                        summary=(
                            f"Adaptation {execution.adaptation_id} version "
                            f"{execution.version} differs from scope {context.version}."
                        ),
                        stage=EvolutionAssuranceLifecycleStage.ADAPTATION,
                        provider_id=self.provider_id,
                        provider_version=self.provider_version,
                        evidence_ref=f"adaptation:{execution.adaptation_id}",
                    )
                )
        return tuple(findings)


def default_evolution_quality_validator_providers() -> tuple[
    LifecycleCompletenessValidator,
    EvidenceAvailabilityValidator,
    VersionConsistencyValidator,
]:
    return (
        LifecycleCompletenessValidator(),
        EvidenceAvailabilityValidator(),
        VersionConsistencyValidator(),
    )


__all__ = [
    "EvidenceAvailabilityValidator",
    "LifecycleCompletenessValidator",
    "VersionConsistencyValidator",
    "default_evolution_quality_validator_providers",
]
