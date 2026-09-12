# © Artur Czarnecki. All rights reserved.

"""W6-D multi-analyzer orchestration — ordering, isolation, aggregation."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.contracts.runtime_intelligence import (
    ANALYZER_OUTCOME_OK,
    ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE,
    AnalyzerExecutionError,
    IntelligenceEvidence,
    IntelligenceEvidenceSourceKind,
    IntelligenceRecommendation,
    IntelligenceRecommendationKind,
    RuntimeIntelligenceAnalyzerPort,
    RuntimeIntelligenceContext,
    RuntimeIntelligenceFactKind,
    RuntimeIntelligenceFactReference,
    RuntimeIntelligenceResult,
)
from intergrax.runtime.runtime_intelligence import (
    DeterministicRuntimeIntelligenceAnalyzer,
    RuntimeIntelligenceAnalyzerOrchestrator,
    RuntimeIntelligenceContextBuilder,
    RuntimeIntelligenceFacts,
    RuntimeIntelligenceOrchestratedAnalysisRequest,
    run_runtime_intelligence_orchestrated_analysis,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_COLLECTED_AT = datetime(2026, 9, 12, 9, 0, tzinfo=UTC)


def _fact_ref() -> RuntimeIntelligenceFactReference:
    return RuntimeIntelligenceFactReference(
        fact_kind=RuntimeIntelligenceFactKind.RUNTIME_EVENT,
        fact_ref="evt_00000000000000000000000000000001",
    )


def _facts() -> RuntimeIntelligenceFacts:
    return RuntimeIntelligenceFacts(
        tenant_id="tenant-a",
        task_id="task_00000000000000000000000000000001",
        run_id="run_00000000000000000000000000000001",
        attempt_id="attempt_00000000000000000000000000000001",
        fact_references=(_fact_ref(),),
        correlation_id="corr-w6d",
        collected_at=_COLLECTED_AT,
    )


def _minimal_result(analyzer_id: str) -> RuntimeIntelligenceResult:
    evidence = IntelligenceEvidence(
        evidence_id=f"ev_{analyzer_id}",
        source_kind=IntelligenceEvidenceSourceKind.RUNTIME_EVENT,
        source_ref="evt_00000000000000000000000000000001",
        relation="supports",
        summary=f"evidence from {analyzer_id}",
    )
    recommendation = IntelligenceRecommendation(
        recommendation_id=f"rec_{analyzer_id}",
        kind=IntelligenceRecommendationKind.EXECUTION_INSIGHT,
        summary=f"recommendation from {analyzer_id}",
    )
    return RuntimeIntelligenceResult(
        analysis_summary=f"analysis from {analyzer_id}",
        confidence=0.7,
        evidence=(evidence,),
        recommendations=(recommendation,),
        analyzer_id=analyzer_id,
        analyzer_version="1.0.0",
    )


class _TaggedAnalyzer:
    def __init__(self, analyzer_id: str) -> None:
        self.analyzer_id = analyzer_id
        self.analyzer_version = "1.0.0"
        self.invoke_order: list[str] = []

    def analyze(self, context: RuntimeIntelligenceContext) -> RuntimeIntelligenceResult:
        self.invoke_order.append(self.analyzer_id)
        return _minimal_result(self.analyzer_id)


class _ExplodingAnalyzer:
    analyzer_id = "plugin.exploding"
    analyzer_version = "1.0.0"

    def analyze(self, context: RuntimeIntelligenceContext) -> RuntimeIntelligenceResult:
        raise AnalyzerExecutionError("simulated orchestration failure")


def test_multiple_analyzers_execute_and_aggregate() -> None:
    first = _TaggedAnalyzer("plugin.alpha")
    second = _TaggedAnalyzer("plugin.beta")
    context = RuntimeIntelligenceContextBuilder().build(_facts())
    result = RuntimeIntelligenceAnalyzerOrchestrator().orchestrate(
        context,
        (first, second),
    )
    assert len(result.outcomes) == 2
    assert all(outcome.outcome == ANALYZER_OUTCOME_OK for outcome in result.outcomes)
    assert result.outcomes[0].result is not None
    assert result.outcomes[1].result is not None
    assert result.outcomes[0].result.analyzer_id == "plugin.alpha"
    assert result.outcomes[1].result.analyzer_id == "plugin.beta"


def test_analyzer_ordering_is_explicit_and_deterministic() -> None:
    beta = _TaggedAnalyzer("plugin.beta")
    alpha = _TaggedAnalyzer("plugin.alpha")
    context = RuntimeIntelligenceContextBuilder().build(_facts())
    orchestrator = RuntimeIntelligenceAnalyzerOrchestrator()

    forward = orchestrator.orchestrate(context, (beta, alpha))
    reverse = orchestrator.orchestrate(context, (alpha, beta))

    assert [o.analyzer_id for o in forward.outcomes] == ["plugin.beta", "plugin.alpha"]
    assert [o.analyzer_id for o in reverse.outcomes] == ["plugin.alpha", "plugin.beta"]
    assert forward.outcomes != reverse.outcomes


def test_failure_isolation_preserves_healthy_analyzer_results() -> None:
    context = RuntimeIntelligenceContextBuilder().build(_facts())
    result = RuntimeIntelligenceAnalyzerOrchestrator().orchestrate(
        context,
        (_ExplodingAnalyzer(), DeterministicRuntimeIntelligenceAnalyzer()),
    )
    assert result.outcomes[0].outcome == ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE
    assert result.outcomes[0].result is None
    assert result.outcomes[1].outcome == ANALYZER_OUTCOME_OK
    assert result.outcomes[1].result is not None
    assert result.outcomes[1].result.analyzer_id == DeterministicRuntimeIntelligenceAnalyzer().analyzer_id


def test_empty_analyzer_collection_yields_no_outcomes() -> None:
    context = RuntimeIntelligenceContextBuilder().build(_facts())
    result = RuntimeIntelligenceAnalyzerOrchestrator().orchestrate(context, ())
    assert result.outcomes == ()
    assert result.context is context


def test_orchestration_preserves_immutable_context() -> None:
    context = RuntimeIntelligenceContextBuilder().build(_facts())
    before_refs = context.fact_references
    RuntimeIntelligenceAnalyzerOrchestrator().orchestrate(
        context,
        (DeterministicRuntimeIntelligenceAnalyzer(), _TaggedAnalyzer("plugin.gamma")),
    )
    assert context.fact_references == before_refs


def test_orchestrated_request_lifecycle_end_to_end() -> None:
    response = run_runtime_intelligence_orchestrated_analysis(
        RuntimeIntelligenceOrchestratedAnalysisRequest(
            facts=_facts(),
            analyzers=(
                DeterministicRuntimeIntelligenceAnalyzer(),
                _TaggedAnalyzer("plugin.delta"),
            ),
        )
    )
    assert response.context.tenant_id == "tenant-a"
    assert len(response.orchestration.outcomes) == 2
    assert response.orchestration.outcomes[0].outcome == ANALYZER_OUTCOME_OK
    assert response.orchestration.outcomes[1].outcome == ANALYZER_OUTCOME_OK


def test_plugins_satisfy_port_without_orchestrator_branching() -> None:
    analyzers: tuple[RuntimeIntelligenceAnalyzerPort, ...] = (
        DeterministicRuntimeIntelligenceAnalyzer(),
        _TaggedAnalyzer("plugin.swap"),
    )
    context = RuntimeIntelligenceContextBuilder().build(_facts())
    result = RuntimeIntelligenceAnalyzerOrchestrator().orchestrate(context, analyzers)
    assert len(result.outcomes) == len(analyzers)
