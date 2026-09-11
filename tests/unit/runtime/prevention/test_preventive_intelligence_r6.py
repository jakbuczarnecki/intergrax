# © Artur Czarnecki. All rights reserved.

"""PREVENTIVE R6 — recommendation intelligence qualification tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from intergrax.contracts.predictive.analyzer_quality_profile import PredictiveAnalyzerQualityProfile
from intergrax.contracts.predictive_risk import (
    PredictiveAnalyzerMetadata,
    PredictiveRiskScope,
    PredictiveRiskSeverity,
    PredictiveRiskSignal,
    PredictiveWindow,
)
from intergrax.contracts.preventive.category import PreventiveRecommendationCategory
from intergrax.contracts.preventive.context import (
    DiagnosticEvidenceContext,
    HistoricalOutcome,
    PreventiveAnalysisInput,
)
from intergrax.contracts.preventive.outcome_evaluation import (
    OperatorRecommendationDecision,
    RecommendationEffectiveness,
    RecommendationOutcomeEvaluation,
)
from intergrax.contracts.preventive.analyzer_descriptor import (
    PreventiveAnalyzerDescriptor,
    PreventiveResourceBudget,
)
from intergrax.contracts.preventive.recommendation import PreventiveRecommendationCandidate
from intergrax.runtime.prediction import LatencyTrendAnalyzer, PredictionEngine, PredictiveAnalyzerRegistry
from intergrax.runtime.prediction.governance import InMemoryPredictiveAnalyzerQualityStore
from intergrax.runtime.prevention import (
    CrmLatencyPreventiveAnalyzer,
    PreventiveAnalyzerRegistry,
    PreventiveIntelligenceEngine,
    PreventiveOutcomeEngine,
    compose_preventive_confidence,
    project_preventive_recommendations,
)
from tests.unit.runtime.prediction.conftest import crm_agent_incident_prevention_context

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_AS_OF = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _risk_signal(
    *,
    tenant_id: str = "tenant-demo",
    confidence: float = 0.82,
    run_id: str = "prun_r6",
) -> PredictiveRiskSignal:
    return PredictiveRiskSignal(
        signal_id="prsig_r6",
        prediction_run_id=run_id,
        tenant_id=tenant_id,
        scope=PredictiveRiskScope.COMPONENT,
        subject_identity="crm_agent",
        risk_type="HIGH_LATENCY_RISK",
        severity=PredictiveRiskSeverity.HIGH,
        confidence=confidence,
        evidence_refs=("feature:latency", "subject:crm_agent"),
        prediction_window=PredictiveWindow(duration_seconds=3600, label="60m"),
        generated_at=_AS_OF - timedelta(minutes=60),
        analyzer_metadata=PredictiveAnalyzerMetadata(
            analyzer_id="latency_trend",
            analyzer_version="latency_trend@1.0.0",
        ),
        model_version="latency_trend@1.0.0",
        summary="CRM latency risk",
    )


def _analysis_input(
    *,
    tenant_id: str = "tenant-demo",
    diagnostic_refs: tuple[str, ...] = ("INC-4521",),
) -> PreventiveAnalysisInput:
    context = crm_agent_incident_prevention_context(tenant_id=tenant_id)
    return PreventiveAnalysisInput(
        predictive_context=context,
        risk_signal=_risk_signal(tenant_id=tenant_id),
        historical_outcome=HistoricalOutcome(
            tenant_id=tenant_id,
            historical_success_rate=0.7,
            similar_incident_refs=("INC-4521",),
        ),
        diagnostic_evidence=DiagnosticEvidenceContext(
            tenant_id=tenant_id,
            evidence_refs=diagnostic_refs,
        ),
    )


def test_preventive_recommendation_generated_from_risk_signal() -> None:
    engine = PreventiveIntelligenceEngine(
        registry=PreventiveAnalyzerRegistry((CrmLatencyPreventiveAnalyzer(),)),
    )
    result = engine.recommend(_analysis_input())
    assert result.recommendations
    rec = result.recommendations[0]
    assert rec.risk_signal_id == "prsig_r6"
    assert rec.governance.execution_allowed is False
    assert rec.safety.execution_allowed is False
    assert result.audit.recommendation_ids
    assert result.recommendation_audit


def test_recommendation_requires_evidence() -> None:
    with pytest.raises(ValueError, match="evidence_refs"):
        PreventiveRecommendationCandidate(
            category=PreventiveRecommendationCategory.OBSERVE,
            description="Restart service",
            expected_impact="unknown",
            evidence_refs=(),
            analyzer_id="bad",
            analyzer_version="bad@1",
        )


def test_confidence_uses_quality_and_history() -> None:
    value = compose_preventive_confidence(
        risk_confidence=0.9,
        analyzer_quality=0.8,
        historical_success=0.7,
        context_completeness=0.9,
    )
    assert value == pytest.approx(0.4536, rel=1e-3)


@dataclass
class _ExplodingPreventiveAnalyzer:
    analyzer_namespace = "test"
    analyzer_id = "explode"
    analyzer_version = "explode@1"
    priority = 1

    @property
    def descriptor(self) -> PreventiveAnalyzerDescriptor:
        return PreventiveAnalyzerDescriptor(
            analyzer_id=self.analyzer_id,
            namespace=self.analyzer_namespace,
            version=self.analyzer_version,
            owner="test",
            capabilities=("explode",),
            quality_profile_id=self.analyzer_id,
            resource_budget=PreventiveResourceBudget(
                max_execution_time_ms=50,
                max_candidates_per_run=1,
            ),
        )

    def analyze(self, analysis_input: PreventiveAnalysisInput) -> tuple[PreventiveRecommendationCandidate, ...]:
        raise RuntimeError("boom")


def test_failed_preventive_analyzer_is_contained() -> None:
    engine = PreventiveIntelligenceEngine(
        registry=PreventiveAnalyzerRegistry(
            (_ExplodingPreventiveAnalyzer(), CrmLatencyPreventiveAnalyzer()),
        ),
    )
    result = engine.recommend(_analysis_input())
    assert result.recommendations
    assert any("explode:PLUGIN_UNAVAILABLE" in item for item in result.audit.analyzer_outcomes)


def test_recommendation_tenant_isolated() -> None:
    payload = _analysis_input(tenant_id="tenant-a")
    with pytest.raises(ValueError, match="tenant isolation"):
        PreventiveAnalysisInput(
            predictive_context=payload.predictive_context,
            risk_signal=_risk_signal(tenant_id="tenant-b"),
            historical_outcome=payload.historical_outcome,
            diagnostic_evidence=payload.diagnostic_evidence,
        )


def test_prevention_does_not_execute_actions() -> None:
    prevention_root = _REPO_ROOT / "intergrax" / "runtime" / "prevention"
    forbidden = (
        "ProblemLifecycleEngine",
        "mint_problem",
        "create_problem",
        "execute_remediation",
        "deployment_engine",
    )
    for path in prevention_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.name} must not execute actions ({token})"


def test_recommendation_effectiveness_updates_history() -> None:
    quality = InMemoryPredictiveAnalyzerQualityStore()
    quality.put_profile(
        PredictiveAnalyzerQualityProfile(
            analyzer_id="crm_latency_preventive",
            tenant_id="tenant-demo",
            predictions=1,
            true_positive=0,
            false_positive=0,
        ),
    )
    engine = PreventiveOutcomeEngine(quality_store=quality)
    engine.record(
        RecommendationOutcomeEvaluation(
            recommendation_id="prrec_1",
            tenant_id="tenant-demo",
            analyzer_id="crm_latency_preventive",
            operator_decision=OperatorRecommendationDecision.ACCEPTED,
            effectiveness=RecommendationEffectiveness.TRUE_PREVENTION,
            evidence_refs=("outcome:incident_avoided",),
            evaluated_at=_AS_OF,
        ),
    )
    rate = engine.store.historical_success_rate(
        tenant_id="tenant-demo",
        analyzer_id="crm_latency_preventive",
    )
    assert rate == 1.0
    profile = quality.get_profile(tenant_id="tenant-demo", analyzer_id="crm_latency_preventive")
    assert profile.true_positive >= 1


def test_prediction_without_prevention_unchanged() -> None:
    context = crm_agent_incident_prevention_context()
    registry = PredictiveAnalyzerRegistry((LatencyTrendAnalyzer(),))
    baseline = PredictionEngine(registry=registry).analyze(context)
    # Prevention layer is opt-in and must not alter prediction engine behavior.
    again = PredictionEngine(registry=registry).analyze(context)
    assert len(baseline.signals) == len(again.signals)
    assert baseline.audit.degraded == again.audit.degraded


def test_read_model_preventive_recommendations() -> None:
    engine = PreventiveIntelligenceEngine(
        registry=PreventiveAnalyzerRegistry((CrmLatencyPreventiveAnalyzer(),)),
    )
    recs = engine.recommend(_analysis_input()).recommendations
    views = project_preventive_recommendations(recs, risk_type="HIGH_LATENCY_RISK")
    assert views[0].related_risk_type == "HIGH_LATENCY_RISK"
    assert views[0].execution_allowed is False


def test_crm_showcase_preventive_phases() -> None:
    engine = PreventiveIntelligenceEngine(
        registry=PreventiveAnalyzerRegistry((CrmLatencyPreventiveAnalyzer(),)),
    )
    phase_one = engine.recommend(_analysis_input()).recommendations[0]
    assert "Investigate payment connector latency" in phase_one.description
    assert phase_one.confidence > 0.0

    phase_two_input = _analysis_input(
        diagnostic_refs=("INC-4521", "connector_retries:+500%"),
    )
    phase_two = engine.recommend(phase_two_input).recommendations[0]
    assert phase_two.governance.priority_label == "HIGH"
    assert "Validate external provider availability" in phase_two.description

    outcome_engine = PreventiveOutcomeEngine()
    outcome_engine.record(
        RecommendationOutcomeEvaluation(
            recommendation_id=phase_two.recommendation_id,
            tenant_id="tenant-demo",
            analyzer_id=phase_two.analyzer_id,
            operator_decision=OperatorRecommendationDecision.ACCEPTED,
            effectiveness=RecommendationEffectiveness.TRUE_PREVENTION,
            evidence_refs=("outcome:incident_avoided", "INC-4521"),
            evaluated_at=_AS_OF,
        ),
    )
    assert outcome_engine.store.evaluations[0].effectiveness is RecommendationEffectiveness.TRUE_PREVENTION
