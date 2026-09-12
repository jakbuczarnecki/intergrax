# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from testing_support.decision_e2e.model_matrix.autonomous_enterprise_adaptation.contracts import (
    AdaptationAuditMetadata,
    AdaptationExecutionResult,
    AdaptationExecutionStatus,
    EvolutionSourceReference,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence import (
    EnterpriseEvolutionIntelligenceEngine,
    EvolutionAnalysisFinding,
    EvolutionGovernanceReference,
    EvolutionHistoryReference,
    EvolutionIntelligenceContext,
    EvolutionIntelligenceDataSourceRef,
    EvolutionIntelligenceRunStatus,
    EvolutionRecommendationKind,
    default_enterprise_evolution_intelligence_engine,
    default_enterprise_evolution_intelligence_provider,
    default_evolution_intelligence_audit_provider,
    default_insight_providers,
    default_metric_providers,
    default_recommendation_providers,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_operations.contracts import (
    EvolutionHealthObservation,
    EvolutionOperationRecord,
    EvolutionOperationStatus,
    EvolutionOperationalEventKind,
)
from testing_support.decision_e2e.model_matrix.self_improvement_governance.contracts import (
    SelfImprovementGovernanceStatus,
)


def _stamp() -> datetime:
    return datetime(2026, 9, 12, 20, 0, 0, tzinfo=UTC)


def _source() -> EvolutionSourceReference:
    return EvolutionSourceReference(
        controlled_evolution_record_id="evo-rec-1",
        proposal_id="prop-routing-1",
        source_insight_ids=("insight-1",),
    )


def _execution_result() -> AdaptationExecutionResult:
    audit = AdaptationAuditMetadata(
        adaptation_task_id="DS-E2E-15J-L13.AUTONOMOUS-ENTERPRISE-ADAPTATION",
        adaptation_layer_version="1",
        adaptation_id="adapt-1",
        adaptation_version="1",
        source_reference=_source(),
        governance_approval_id="gov-appr-1",
        approver_identity="enterprise-governance-board",
        provider_id="default_enterprise_adaptation",
        provider_version="1",
        applied_change_reference="adapted:adapt-1:v1:prop-routing-1",
        outcome_status=AdaptationExecutionStatus.APPLIED,
        executed_at=_stamp(),
        outcome_summary="Applied governed adaptation.",
    )
    return AdaptationExecutionResult(
        status=AdaptationExecutionStatus.APPLIED,
        provider_id="default_enterprise_adaptation",
        provider_version="1",
        applied_change_reference="adapted:adapt-1:v1:prop-routing-1",
        audit_metadata=audit,
        adaptation_id="adapt-1",
        version="1",
        source_reference=_source(),
    )


def _operation_record() -> EvolutionOperationRecord:
    return EvolutionOperationRecord(
        record_id="ops-rec-1",
        adaptation_id="adapt-1",
        version="1",
        event_kind=EvolutionOperationalEventKind.OBSERVED,
        operational_status=EvolutionOperationStatus.ACTIVE,
        occurred_at=_stamp(),
        operator_identity="ops-analyst-1",
        summary="Routine observation.",
    )


def _health_observation() -> EvolutionHealthObservation:
    return EvolutionHealthObservation(
        adaptation_id="adapt-1",
        version="1",
        operational_status=EvolutionOperationStatus.ACTIVE,
        quality_indicator="stable",
        risk_indicator="low",
        summary="Stable health.",
    )


def _context() -> EvolutionIntelligenceContext:
    return EvolutionIntelligenceContext(
        adaptation_id="adapt-1",
        version="1",
        history_reference=EvolutionHistoryReference(
            adaptation_id="adapt-1",
            version="1",
            proposal_id="prop-routing-1",
            controlled_evolution_record_id="evo-rec-1",
        ),
        operation_records=(_operation_record(),),
        execution_results=(_execution_result(),),
        health_observations=(_health_observation(),),
        controlled_evolution=None,
        governance_reference=EvolutionGovernanceReference(
            governance_decision_reference="gov-decision-1",
            governance_status=SelfImprovementGovernanceStatus.APPROVED,
            approval_id="gov-appr-1",
        ),
        governance_decision=None,
    )


def test_default_intelligence_provider_analyzes_adaptation_history() -> None:
    provider = default_enterprise_evolution_intelligence_provider()
    result = provider.analyze(_context(), analyzed_at=_stamp())
    assert result.status is EvolutionIntelligenceRunStatus.COMPLETE
    assert result.analysis_findings
    assert any(
        item.analyzer_id == "adaptation_effectiveness_analyzer"
        for item in result.analysis_findings
    )


@dataclass(frozen=True, slots=True)
class CustomEvolutionAnalyzerProvider:
    @property
    def provider_id(self) -> str:
        return "custom_evolution_analyzer"

    @property
    def provider_version(self) -> str:
        return "9"

    def analyze(
        self,
        context: EvolutionIntelligenceContext,
    ) -> tuple[EvolutionAnalysisFinding, ...]:
        return (
            EvolutionAnalysisFinding(
                finding_id="finding:custom",
                analyzer_id=self.provider_id,
                analyzer_version=self.provider_version,
                category="custom",
                summary=f"Custom analysis for {context.adaptation_id}.",
                data_source_refs=(
                    EvolutionIntelligenceDataSourceRef(
                        source_id="custom:source",
                        source_kind="custom",
                        description="Custom analyzer source.",
                    ),
                ),
            ),
        )


def test_analyzer_swap_without_engine_change() -> None:
    engine = EnterpriseEvolutionIntelligenceEngine(
        intelligence_providers=(),
        analyzer_providers=(CustomEvolutionAnalyzerProvider(),),
        metric_providers=default_metric_providers(),
        insight_providers=default_insight_providers(),
        recommendation_providers=default_recommendation_providers(),
        audit_provider=default_evolution_intelligence_audit_provider(),
    )
    result = engine.analyze(_context(), analyzed_at=_stamp())
    assert any(
        item.analyzer_id == "custom_evolution_analyzer"
        for item in result.analysis_findings
    )


def test_insight_includes_evidence_sources() -> None:
    engine = default_enterprise_evolution_intelligence_engine()
    result = engine.analyze(_context(), analyzed_at=_stamp())
    assert result.insights
    insight = result.insights[0]
    assert insight.evidence_source_refs
    assert insight.linked_finding_ids


def test_recommendation_provider_generates_recommendation() -> None:
    engine = default_enterprise_evolution_intelligence_engine()
    result = engine.analyze(_context(), analyzed_at=_stamp())
    assert result.recommendations
    recommendation = result.recommendations[0]
    assert recommendation.kind in (
        EvolutionRecommendationKind.CONTINUE_OBSERVATION,
        EvolutionRecommendationKind.REVIEW_ADAPTATION,
        EvolutionRecommendationKind.COLLECT_MORE_EVIDENCE,
    )
    assert recommendation.linked_insight_ids


def test_intelligence_audit_records_analyzers_providers_sources_timestamp() -> None:
    engine = default_enterprise_evolution_intelligence_engine()
    result = engine.analyze(_context(), analyzed_at=_stamp())
    audit = result.audit
    assert audit.analyzer_ids
    assert "adaptation_effectiveness_analyzer" in audit.analyzer_ids
    assert audit.insight_provider_ids
    assert audit.recommendation_provider_ids
    assert audit.data_source_refs
    assert audit.analyzed_at == _stamp()
    assert audit.adaptation_id == "adapt-1"
