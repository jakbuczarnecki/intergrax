# © Artur Czarnecki. All rights reserved.

"""W6-C context builder, deterministic analyzer, and request lifecycle tests."""

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
    InvalidIntelligenceContextError,
    RuntimeIntelligenceAnalyzerPort,
    RuntimeIntelligenceContext,
    RuntimeIntelligenceFactKind,
    RuntimeIntelligenceFactReference,
    RuntimeIntelligenceResult,
    run_runtime_intelligence_analyzer_isolated,
    validate_runtime_intelligence_context,
)
from intergrax.runtime.runtime_intelligence import (
    DeterministicRuntimeIntelligenceAnalyzer,
    RuntimeIntelligenceAnalysisRequest,
    RuntimeIntelligenceContextBuilder,
    RuntimeIntelligenceFacts,
    RuntimeIntelligenceObservedSignal,
    RuntimeIntelligenceSignalKind,
    run_runtime_intelligence_analysis,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_COLLECTED_AT = datetime(2026, 9, 12, 8, 0, tzinfo=UTC)


def _fact_ref(kind: RuntimeIntelligenceFactKind = RuntimeIntelligenceFactKind.RUNTIME_EVENT) -> RuntimeIntelligenceFactReference:
    return RuntimeIntelligenceFactReference(
        fact_kind=kind,
        fact_ref="evt_00000000000000000000000000000001",
    )


def _facts(
    *,
    fact_refs: tuple[RuntimeIntelligenceFactReference, ...] | None = None,
    signals: tuple[RuntimeIntelligenceObservedSignal, ...] = (),
) -> RuntimeIntelligenceFacts:
    return RuntimeIntelligenceFacts(
        tenant_id="tenant-a",
        task_id="task_00000000000000000000000000000001",
        run_id="run_00000000000000000000000000000001",
        attempt_id="attempt_00000000000000000000000000000001",
        fact_references=fact_refs or (_fact_ref(),),
        correlation_id="corr-w6c",
        collected_at=_COLLECTED_AT,
        observed_signals=signals,
    )


def test_builder_creates_valid_context() -> None:
    facts = _facts(
        signals=(
            RuntimeIntelligenceObservedSignal(
                kind=RuntimeIntelligenceSignalKind.RETRY_PRESSURE,
                intensity=0.72,
                source_fact_ref="evt_00000000000000000000000000000001",
            ),
        ),
    )
    context = RuntimeIntelligenceContextBuilder().build(facts)
    validate_runtime_intelligence_context(context)
    assert context.tenant_id == facts.tenant_id
    assert len(context.fact_references) == 2
    assert any(ref.fact_ref.startswith("intelligence_signal:") for ref in context.fact_references)


def test_builder_rejects_empty_fact_projection() -> None:
    facts = RuntimeIntelligenceFacts(
        tenant_id="tenant-a",
        task_id="task_00000000000000000000000000000001",
        run_id="run_00000000000000000000000000000001",
        fact_references=(),
        correlation_id="corr-empty",
        collected_at=_COLLECTED_AT,
    )
    with pytest.raises(InvalidIntelligenceContextError):
        RuntimeIntelligenceContextBuilder().build(facts)


def test_deterministic_analyzer_satisfies_port() -> None:
    analyzer = DeterministicRuntimeIntelligenceAnalyzer()
    assert isinstance(analyzer, RuntimeIntelligenceAnalyzerPort)


def test_deterministic_analyzer_same_context_same_result() -> None:
    facts = _facts(
        signals=(
            RuntimeIntelligenceObservedSignal(
                kind=RuntimeIntelligenceSignalKind.REPEATED_FAILURES,
                intensity=0.81,
                source_fact_ref="evt_00000000000000000000000000000001",
            ),
        ),
    )
    context = RuntimeIntelligenceContextBuilder().build(facts)
    analyzer = DeterministicRuntimeIntelligenceAnalyzer()
    first = analyzer.analyze(context)
    second = analyzer.analyze(context)
    assert first == second


def test_analyzer_does_not_mutate_context_or_execute_side_effects() -> None:
    facts = _facts()
    context = RuntimeIntelligenceContextBuilder().build(facts)
    before_refs = context.fact_references
    analyzer = DeterministicRuntimeIntelligenceAnalyzer()
    result = analyzer.analyze(context)
    assert context.fact_references == before_refs
    assert result.evidence
    assert result.recommendations


class _AlternateAnalyzer:
    analyzer_id = "plugin.alternate"
    analyzer_version = "0.1.0"

    def analyze(self, context: RuntimeIntelligenceContext) -> RuntimeIntelligenceResult:
        evidence = IntelligenceEvidence(
            evidence_id="ev_alt",
            source_kind=IntelligenceEvidenceSourceKind.RUNTIME_EVENT,
            source_ref=context.fact_references[0].fact_ref,
            relation="supports",
            summary="alternate plugin",
        )
        recommendation = IntelligenceRecommendation(
            recommendation_id="rec_alt",
            kind=IntelligenceRecommendationKind.EXECUTION_INSIGHT,
            summary="Alternate analyzer recommendation",
        )
        return RuntimeIntelligenceResult(
            analysis_summary="Alternate plugin analysis",
            confidence=0.6,
            evidence=(evidence,),
            recommendations=(recommendation,),
            analyzer_id=self.analyzer_id,
            analyzer_version=self.analyzer_version,
        )


def test_plugin_analyzer_swap_at_request_boundary() -> None:
    facts = _facts()
    deterministic = run_runtime_intelligence_analysis(
        RuntimeIntelligenceAnalysisRequest(facts=facts, analyzer=DeterministicRuntimeIntelligenceAnalyzer())
    )
    alternate = run_runtime_intelligence_analysis(
        RuntimeIntelligenceAnalysisRequest(facts=facts, analyzer=_AlternateAnalyzer())
    )
    assert deterministic.outcome.result is not None
    assert alternate.outcome.result is not None
    assert deterministic.outcome.result.analyzer_id != alternate.outcome.result.analyzer_id


def test_request_lifecycle_invalid_context_isolated() -> None:
    facts = RuntimeIntelligenceFacts(
        tenant_id="tenant-a",
        task_id="task_00000000000000000000000000000001",
        run_id="run_00000000000000000000000000000001",
        fact_references=(),
        correlation_id="corr-invalid",
        collected_at=_COLLECTED_AT,
    )
    with pytest.raises(InvalidIntelligenceContextError):
        run_runtime_intelligence_analysis(
            RuntimeIntelligenceAnalysisRequest(
                facts=facts,
                analyzer=DeterministicRuntimeIntelligenceAnalyzer(),
            )
        )


def test_failure_isolation_analyzer_error_does_not_abort_execution_simulation() -> None:
    class _ExplodingAnalyzer:
        analyzer_id = "runtime_intelligence.deterministic"
        analyzer_version = "1.0.0"

        def analyze(self, context: RuntimeIntelligenceContext) -> RuntimeIntelligenceResult:
            raise AnalyzerExecutionError("simulated plugin failure")

    facts = _facts()
    context = RuntimeIntelligenceContextBuilder().build(facts)
    execution_steps: list[str] = []
    outcome = run_runtime_intelligence_analyzer_isolated(_ExplodingAnalyzer(), context)
    if outcome.outcome == ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE:
        execution_steps.append("intelligence_degraded")
    outcome_ok = run_runtime_intelligence_analyzer_isolated(
        DeterministicRuntimeIntelligenceAnalyzer(),
        context,
    )
    if outcome_ok.outcome == ANALYZER_OUTCOME_OK:
        execution_steps.append("intelligence_ok")
    execution_steps.append("execution_continued")
    assert execution_steps == ["intelligence_degraded", "intelligence_ok", "execution_continued"]


def test_end_to_end_request_success() -> None:
    facts = _facts(
        signals=(
            RuntimeIntelligenceObservedSignal(
                kind=RuntimeIntelligenceSignalKind.RESOURCE_PRESSURE,
                intensity=0.66,
                source_fact_ref="evt_00000000000000000000000000000001",
            ),
        ),
    )
    response = run_runtime_intelligence_analysis(
        RuntimeIntelligenceAnalysisRequest(
            facts=facts,
            analyzer=DeterministicRuntimeIntelligenceAnalyzer(),
        )
    )
    assert response.outcome.outcome == ANALYZER_OUTCOME_OK
    assert response.outcome.result is not None
    assert response.outcome.result.evidence
