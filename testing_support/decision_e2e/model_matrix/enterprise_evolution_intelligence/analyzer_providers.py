# © Artur Czarnecki. All rights reserved.

"""Evolution analyzer provider plugins (DS-E2E-15J-L15)."""

from __future__ import annotations

from dataclasses import dataclass

from testing_support.decision_e2e.model_matrix.autonomous_enterprise_adaptation.contracts import (
    AdaptationExecutionStatus,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.contracts import (
    EvolutionAnalysisFinding,
    EvolutionIntelligenceContext,
    EvolutionIntelligenceDataSourceRef,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_operations.contracts import (
    EvolutionOperationStatus,
)
from testing_support.decision_e2e.model_matrix.self_improvement_governance.contracts import (
    SelfImprovementGovernanceStatus,
)

_ANALYZER_VERSION = "1"


def _history_ref(
    context: EvolutionIntelligenceContext,
) -> EvolutionIntelligenceDataSourceRef:
    ref = context.history_reference
    return EvolutionIntelligenceDataSourceRef(
        source_id=f"history:{ref.adaptation_id}:{ref.version}",
        source_kind="evolution_history",
        description="Evolution history reference for adaptation under analysis.",
    )


@dataclass(frozen=True, slots=True)
class AdaptationEffectivenessAnalyzer:
    @property
    def provider_id(self) -> str:
        return "adaptation_effectiveness_analyzer"

    @property
    def provider_version(self) -> str:
        return _ANALYZER_VERSION

    def analyze(
        self,
        context: EvolutionIntelligenceContext,
    ) -> tuple[EvolutionAnalysisFinding, ...]:
        applied = sum(
            1
            for item in context.execution_results
            if item.status is AdaptationExecutionStatus.APPLIED
        )
        stable_health = sum(
            1
            for item in context.health_observations
            if item.quality_indicator == "stable"
        )
        effective = (
            applied > 0 and stable_health >= len(context.health_observations) // 2
        )
        summary = (
            f"Adaptation {context.adaptation_id} shows effective rollout "
            f"(applied={applied}, stable_observations={stable_health})."
            if effective
            else f"Adaptation {context.adaptation_id} effectiveness is uncertain "
            f"(applied={applied}, stable_observations={stable_health})."
        )
        refs = (_history_ref(context),)
        if context.execution_results:
            refs = (
                *refs,
                EvolutionIntelligenceDataSourceRef(
                    source_id=f"execution:{context.execution_results[0].adaptation_id}",
                    source_kind="adaptation_execution",
                    description="Adaptation execution outcome history.",
                ),
            )
        return (
            EvolutionAnalysisFinding(
                finding_id=f"finding:effectiveness:{context.adaptation_id}",
                analyzer_id=self.provider_id,
                analyzer_version=self.provider_version,
                category="effectiveness",
                summary=summary,
                data_source_refs=refs,
            ),
        )


@dataclass(frozen=True, slots=True)
class CostImpactAnalyzer:
    @property
    def provider_id(self) -> str:
        return "cost_impact_analyzer"

    @property
    def provider_version(self) -> str:
        return _ANALYZER_VERSION

    def analyze(
        self,
        context: EvolutionIntelligenceContext,
    ) -> tuple[EvolutionAnalysisFinding, ...]:
        operation_count = len(context.operation_records)
        elevated_risk = any(
            item.risk_indicator in ("medium", "high")
            for item in context.health_observations
        )
        cost_signal = "elevated" if elevated_risk or operation_count > 5 else "nominal"
        summary = (
            f"Cost impact signal for {context.adaptation_id} is {cost_signal} "
            f"(operations={operation_count})."
        )
        return (
            EvolutionAnalysisFinding(
                finding_id=f"finding:cost:{context.adaptation_id}",
                analyzer_id=self.provider_id,
                analyzer_version=self.provider_version,
                category="cost",
                summary=summary,
                data_source_refs=(_history_ref(context),),
            ),
        )


@dataclass(frozen=True, slots=True)
class StabilityAnalyzer:
    @property
    def provider_id(self) -> str:
        return "stability_analyzer"

    @property
    def provider_version(self) -> str:
        return _ANALYZER_VERSION

    def analyze(
        self,
        context: EvolutionIntelligenceContext,
    ) -> tuple[EvolutionAnalysisFinding, ...]:
        failed_ops = sum(
            1
            for item in context.operation_records
            if item.operational_status is EvolutionOperationStatus.FAILED
        )
        degraded = any(
            item.quality_indicator == "degraded" for item in context.health_observations
        )
        stable = failed_ops == 0 and not degraded
        summary = (
            f"Stability for {context.adaptation_id} is stable."
            if stable
            else f"Stability concerns for {context.adaptation_id} "
            f"(failed_operations={failed_ops}, degraded_health={degraded})."
        )
        refs: tuple[EvolutionIntelligenceDataSourceRef, ...] = (_history_ref(context),)
        if context.health_observations:
            obs = context.health_observations[0]
            refs = (
                *refs,
                EvolutionIntelligenceDataSourceRef(
                    source_id=f"health:{obs.adaptation_id}:{obs.version}",
                    source_kind="health_observation",
                    description="Operational health observation sample.",
                ),
            )
        return (
            EvolutionAnalysisFinding(
                finding_id=f"finding:stability:{context.adaptation_id}",
                analyzer_id=self.provider_id,
                analyzer_version=self.provider_version,
                category="stability",
                summary=summary,
                data_source_refs=refs,
            ),
        )


@dataclass(frozen=True, slots=True)
class GovernanceComplianceAnalyzer:
    @property
    def provider_id(self) -> str:
        return "governance_compliance_analyzer"

    @property
    def provider_version(self) -> str:
        return _ANALYZER_VERSION

    def analyze(
        self,
        context: EvolutionIntelligenceContext,
    ) -> tuple[EvolutionAnalysisFinding, ...]:
        gov = context.governance_reference
        compliant = (
            gov is not None
            and gov.governance_status is SelfImprovementGovernanceStatus.APPROVED
        )
        summary = (
            "Governance trail indicates approved controlled evolution."
            if compliant
            else "Governance trail missing or not fully approved for this adaptation."
        )
        refs: tuple[EvolutionIntelligenceDataSourceRef, ...] = (_history_ref(context),)
        if gov is not None:
            refs = (
                *refs,
                EvolutionIntelligenceDataSourceRef(
                    source_id=gov.governance_decision_reference,
                    source_kind="governance_reference",
                    description="Governance decision reference for adaptation.",
                ),
            )
        return (
            EvolutionAnalysisFinding(
                finding_id=f"finding:governance:{context.adaptation_id}",
                analyzer_id=self.provider_id,
                analyzer_version=self.provider_version,
                category="governance",
                summary=summary,
                data_source_refs=refs,
            ),
        )


def default_analyzer_providers() -> tuple[
    AdaptationEffectivenessAnalyzer,
    CostImpactAnalyzer,
    StabilityAnalyzer,
    GovernanceComplianceAnalyzer,
]:
    return (
        AdaptationEffectivenessAnalyzer(),
        CostImpactAnalyzer(),
        StabilityAnalyzer(),
        GovernanceComplianceAnalyzer(),
    )


__all__ = [
    "AdaptationEffectivenessAnalyzer",
    "CostImpactAnalyzer",
    "GovernanceComplianceAnalyzer",
    "StabilityAnalyzer",
    "default_analyzer_providers",
]
