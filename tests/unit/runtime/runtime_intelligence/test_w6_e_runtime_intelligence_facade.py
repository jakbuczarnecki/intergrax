# © Artur Czarnecki. All rights reserved.

"""W6-E Runtime Intelligence facade — delegation, ordering, isolation."""

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
    RuntimeIntelligenceAnalysisRequest,
    RuntimeIntelligenceAnalyzerOrchestrationResult,
    RuntimeIntelligenceAnalyzerOrchestrator,
    RuntimeIntelligenceFacade,
    RuntimeIntelligenceFacts,
    RuntimeIntelligenceOrchestratedAnalysisRequest,
    run_runtime_intelligence_analysis,
    run_runtime_intelligence_orchestrated_analysis,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_COLLECTED_AT = datetime(2026, 9, 12, 10, 0, tzinfo=UTC)


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
        correlation_id="corr-w6e",
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

    def analyze(self, context: RuntimeIntelligenceContext) -> RuntimeIntelligenceResult:
        return _minimal_result(self.analyzer_id)


class _ExplodingAnalyzer:
    analyzer_id = "plugin.exploding"
    analyzer_version = "1.0.0"

    def analyze(self, context: RuntimeIntelligenceContext) -> RuntimeIntelligenceResult:
        raise AnalyzerExecutionError("simulated orchestration failure")


class _RecordingOrchestrator:
    __slots__ = ("_inner", "invoked")

    def __init__(self) -> None:
        self._inner = RuntimeIntelligenceAnalyzerOrchestrator()
        self.invoked = False

    def orchestrate(
        self,
        context: RuntimeIntelligenceContext,
        analyzers: tuple[RuntimeIntelligenceAnalyzerPort, ...],
    ) -> RuntimeIntelligenceAnalyzerOrchestrationResult:
        self.invoked = True
        return self._inner.orchestrate(context, analyzers)


def test_facade_single_analysis_matches_direct_lifecycle() -> None:
    facts = _facts()
    analyzer = DeterministicRuntimeIntelligenceAnalyzer()
    expected = run_runtime_intelligence_analysis(
        RuntimeIntelligenceAnalysisRequest(facts=facts, analyzer=analyzer)
    )
    actual = RuntimeIntelligenceFacade().analyze(facts, analyzer)
    assert actual.context == expected.context
    assert actual.outcome == expected.outcome


def test_facade_orchestrated_matches_direct_lifecycle() -> None:
    facts = _facts()
    analyzers: tuple[RuntimeIntelligenceAnalyzerPort, ...] = (
        DeterministicRuntimeIntelligenceAnalyzer(),
        _TaggedAnalyzer("plugin.facade"),
    )
    expected = run_runtime_intelligence_orchestrated_analysis(
        RuntimeIntelligenceOrchestratedAnalysisRequest(facts=facts, analyzers=analyzers)
    )
    actual = RuntimeIntelligenceFacade().analyze_orchestrated(facts, analyzers)
    assert actual.context == expected.context
    assert actual.orchestration == expected.orchestration


def test_facade_orchestrated_uses_injected_orchestrator() -> None:
    orchestrator = _RecordingOrchestrator()
    facade = RuntimeIntelligenceFacade(orchestrator=orchestrator)
    facade.analyze_orchestrated(_facts(), (DeterministicRuntimeIntelligenceAnalyzer(),))
    assert orchestrator.invoked is True


def test_facade_analyzer_ordering_is_explicit_and_deterministic() -> None:
    facade = RuntimeIntelligenceFacade()
    beta = _TaggedAnalyzer("plugin.beta")
    alpha = _TaggedAnalyzer("plugin.alpha")

    forward = facade.analyze_orchestrated(_facts(), (beta, alpha))
    reverse = facade.analyze_orchestrated(_facts(), (alpha, beta))

    assert [o.analyzer_id for o in forward.orchestration.outcomes] == ["plugin.beta", "plugin.alpha"]
    assert [o.analyzer_id for o in reverse.orchestration.outcomes] == ["plugin.alpha", "plugin.beta"]


def test_facade_failure_isolation_preserves_healthy_analyzer_results() -> None:
    response = RuntimeIntelligenceFacade().analyze_orchestrated(
        _facts(),
        (_ExplodingAnalyzer(), DeterministicRuntimeIntelligenceAnalyzer()),
    )
    outcomes = response.orchestration.outcomes
    assert outcomes[0].outcome == ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE
    assert outcomes[0].result is None
    assert outcomes[1].outcome == ANALYZER_OUTCOME_OK
    assert outcomes[1].result is not None


def test_facade_preserves_immutable_context() -> None:
    response = RuntimeIntelligenceFacade().analyze_orchestrated(
        _facts(),
        (
            DeterministicRuntimeIntelligenceAnalyzer(),
            _TaggedAnalyzer("plugin.gamma"),
        ),
    )
    before_refs = response.context.fact_references
    second = RuntimeIntelligenceFacade().analyze_orchestrated(
        _facts(),
        (_TaggedAnalyzer("plugin.delta"),),
    )
    assert response.context.fact_references == before_refs
    assert second.context.fact_references == before_refs
