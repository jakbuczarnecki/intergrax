# © Artur Czarnecki. All rights reserved.

"""W6-B Runtime Intelligence contract conformance and isolation tests."""

from __future__ import annotations

import dataclasses
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
    RuntimeIntelligenceContextMetadata,
    RuntimeIntelligenceFactKind,
    RuntimeIntelligenceFactReference,
    RuntimeIntelligenceResult,
    run_runtime_intelligence_analyzer_isolated,
    validate_runtime_intelligence_context,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _context(
    *,
    fact_refs: tuple[RuntimeIntelligenceFactReference, ...] | None = None,
    collected_at: datetime | None = None,
) -> RuntimeIntelligenceContext:
    if fact_refs is None:
        refs = (
            RuntimeIntelligenceFactReference(
                fact_kind=RuntimeIntelligenceFactKind.RUNTIME_EVENT,
                fact_ref="evt_00000000000000000000000000000001",
            ),
        )
    else:
        refs = fact_refs
    return RuntimeIntelligenceContext(
        tenant_id="tenant-a",
        task_id="task_00000000000000000000000000000001",
        run_id="run_00000000000000000000000000000001",
        attempt_id="attempt_00000000000000000000000000000001",
        fact_references=refs,
        metadata=RuntimeIntelligenceContextMetadata(
            collected_at=collected_at or datetime(2026, 9, 12, 8, 0, tzinfo=UTC),
            correlation_id="corr-1",
        ),
    )


def _minimal_result(analyzer_id: str = "fake.local") -> RuntimeIntelligenceResult:
    evidence = IntelligenceEvidence(
        evidence_id="ev_1",
        source_kind=IntelligenceEvidenceSourceKind.RUNTIME_EVENT,
        source_ref="evt_00000000000000000000000000000001",
        relation="supports",
        summary="observed terminal failure event",
    )
    recommendation = IntelligenceRecommendation(
        recommendation_id="rec_1",
        kind=IntelligenceRecommendationKind.EXECUTION_INSIGHT,
        summary="Review dependency admission before retry",
    )
    return RuntimeIntelligenceResult(
        analysis_summary="Run failed after dependency timeout",
        confidence=0.82,
        evidence=(evidence,),
        recommendations=(recommendation,),
        analyzer_id=analyzer_id,
        analyzer_version="1.0.0",
    )


class _FakeLocalAnalyzer:
    analyzer_id = "fake.local"
    analyzer_version = "1.0.0"

    def analyze(self, context: RuntimeIntelligenceContext) -> RuntimeIntelligenceResult:
        return _minimal_result(self.analyzer_id)


class _FakeMlAnalyzer:
    analyzer_id = "fake.ml"
    analyzer_version = "2.1.0"

    def analyze(self, context: RuntimeIntelligenceContext) -> RuntimeIntelligenceResult:
        return _minimal_result(self.analyzer_id)


class _FakeExternalAnalyzer:
    analyzer_id = "fake.external"
    analyzer_version = "0.9.0"

    def analyze(self, context: RuntimeIntelligenceContext) -> RuntimeIntelligenceResult:
        return _minimal_result(self.analyzer_id)


class _ExplodingAnalyzer:
    analyzer_id = "fake.explode"
    analyzer_version = "1.0.0"

    def analyze(self, context: RuntimeIntelligenceContext) -> RuntimeIntelligenceResult:
        raise AnalyzerExecutionError("model unavailable")


def test_analyzer_port_runtime_checkable() -> None:
    assert isinstance(_FakeLocalAnalyzer(), RuntimeIntelligenceAnalyzerPort)
    assert isinstance(_FakeMlAnalyzer(), RuntimeIntelligenceAnalyzerPort)
    assert isinstance(_FakeExternalAnalyzer(), RuntimeIntelligenceAnalyzerPort)


def test_plugin_fake_analyzers_return_conformant_results() -> None:
    context = _context()
    for analyzer in (_FakeLocalAnalyzer(), _FakeMlAnalyzer(), _FakeExternalAnalyzer()):
        outcome = run_runtime_intelligence_analyzer_isolated(analyzer, context)
        assert outcome.outcome == ANALYZER_OUTCOME_OK
        assert outcome.result is not None
        assert outcome.result.analyzer_id == analyzer.analyzer_id
        assert outcome.result.recommendations[0].kind is IntelligenceRecommendationKind.EXECUTION_INSIGHT


def test_models_are_immutable() -> None:
    context = _context()
    result = _minimal_result()
    evidence = result.evidence[0]
    recommendation = result.recommendations[0]
    for obj, field, value in (
        (context, "tenant_id", "other"),
        (result, "confidence", 0.5),
        (evidence, "source_ref", "x"),
        (recommendation, "summary", "mutated"),
    ):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, field, value)


def test_validate_context_requires_fact_refs() -> None:
    empty_facts = _context(fact_refs=())
    with pytest.raises(InvalidIntelligenceContextError):
        validate_runtime_intelligence_context(empty_facts)


def test_isolated_invoke_invalid_context_does_not_raise() -> None:
    outcome = run_runtime_intelligence_analyzer_isolated(_FakeLocalAnalyzer(), _context(fact_refs=()))
    assert outcome.result is None
    assert outcome.outcome != ANALYZER_OUTCOME_OK


def test_failure_isolation_analyzer_error_does_not_abort_execution_simulation() -> None:
    """Execution hot path continues when intelligence is unavailable."""
    context = _context()
    analyzers: tuple[RuntimeIntelligenceAnalyzerPort, ...] = (
        _ExplodingAnalyzer(),
        _FakeLocalAnalyzer(),
    )
    execution_steps: list[str] = []
    for analyzer in analyzers:
        outcome = run_runtime_intelligence_analyzer_isolated(analyzer, context)
        if outcome.outcome == ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE:
            execution_steps.append("intelligence_degraded")
            continue
        execution_steps.append("intelligence_ok")
    execution_steps.append("execution_continued")
    assert execution_steps == ["intelligence_degraded", "intelligence_ok", "execution_continued"]


def test_analyzer_identity_mismatch_treated_as_plugin_unavailable() -> None:
    class _WrongIdAnalyzer:
        analyzer_id = "expected.id"
        analyzer_version = "1.0.0"

        def analyze(self, context: RuntimeIntelligenceContext) -> RuntimeIntelligenceResult:
            return _minimal_result("other.id")

    outcome = run_runtime_intelligence_analyzer_isolated(_WrongIdAnalyzer(), _context())
    assert outcome.outcome == ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE
    assert outcome.result is None


def test_result_rejects_empty_evidence() -> None:
    with pytest.raises(ValueError, match="evidence must be non-empty"):
        RuntimeIntelligenceResult(
            analysis_summary="x",
            confidence=0.5,
            evidence=(),
            recommendations=_minimal_result().recommendations,
            analyzer_id="a",
            analyzer_version="1",
        )
