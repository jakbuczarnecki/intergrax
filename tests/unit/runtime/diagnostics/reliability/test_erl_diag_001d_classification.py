# © Artur Czarnecki. All rights reserved.

"""ERL-DIAG-001D — severity and recommendation strategy pluginability."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.contracts.enterprise_reliability.case_lifecycle import (
    ReliabilityCaseLifecycleState,
)
from intergrax.contracts.enterprise_reliability.diagnostics import (
    AutomationSafetyHint,
    ExternalEffectReliabilityDiagnosticSeverity,
    ExternalEffectReliabilityObservation,
    ExternalEffectReliabilityOperatorRecommendationKind,
    ExternalEffectReliabilitySeverityDecision,
    ExternalEffectReliabilitySignalKind,
    ReliabilityDiagnosticArtifactRefs,
    ReliabilityDiagnosticCorrelation,
    ReliabilityDiagnosticRecommendationStrategyVersion,
    ReliabilityDiagnosticSeverityStrategyId,
    build_reliability_diagnostic_classification_context,
)
from intergrax.runtime.diagnostics.reliability.conservative_reliability_recommendation_strategy import (
    ConservativeReliabilityRecommendationStrategy,
)
from intergrax.runtime.diagnostics.reliability.conservative_reliability_severity_strategy import (
    ConservativeReliabilitySeverityStrategy,
)
from intergrax.runtime.diagnostics.reliability.observation_to_problem_signal import (
    map_handoff_to_platform_problem_signal,
)
from intergrax.runtime.diagnostics.reliability.reliability_diagnostic_classification_service import (
    ReliabilityDiagnosticClassificationService,
    ReliabilityDiagnosticClassificationValidationError,
    validate_severity_decision,
)
from intergrax.runtime.diagnostics.reliability.reliability_diagnostic_handoff import (
    map_observation_to_handoff,
)
from intergrax.runtime.diagnostics.reliability.reliability_diagnostic_strategy_composition import (
    build_reliability_diagnostic_classification_service,
)
from intergrax.runtime.observability.problem_signal import PROBLEM_SEVERITY_ERROR
from tests.unit.erl_diagnostics_plugins.critical_truth_unavailable_severity_strategy import (
    CriticalTruthUnavailableSeverityStrategy,
)
from tests.unit.erl_diagnostics_plugins.escalate_on_critical_recommendation_strategy import (
    EscalateOnCriticalRecommendationStrategy,
)

pytestmark = pytest.mark.unit

_RECORDED_AT = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)


def _observation(
    signal_kind: ExternalEffectReliabilitySignalKind = (
        ExternalEffectReliabilitySignalKind.TRUTH_UNAVAILABLE
    ),
) -> ExternalEffectReliabilityObservation:
    return ExternalEffectReliabilityObservation(
        observation_id="obs-001d",
        tenant_id="tenant-001d",
        signal_kind=signal_kind,
        recorded_at=_RECORDED_AT,
        reliability_case_id="case-001d",
        correlation=ReliabilityDiagnosticCorrelation(
            tenant_id="tenant-001d",
            correlation_id="corr-001d",
            reliability_case_id="case-001d",
            external_effect_contract_id="contract-1",
        ),
        lifecycle_state=ReliabilityCaseLifecycleState.UNKNOWN_DETECTED,
        artifact_refs=ReliabilityDiagnosticArtifactRefs(evidence_ref="evidence-1"),
        execution_safety_hint=AutomationSafetyHint.SAFE,
    )


def _context(
    signal_kind: ExternalEffectReliabilitySignalKind = (
        ExternalEffectReliabilitySignalKind.TRUTH_UNAVAILABLE
    ),
):
    return build_reliability_diagnostic_classification_context(_observation(signal_kind))


def test_default_severity_strategy_deterministic() -> None:
    strategy = ConservativeReliabilitySeverityStrategy()
    first = strategy.classify(_context())
    second = strategy.classify(_context())
    assert first == second
    assert first.severity is ExternalEffectReliabilityDiagnosticSeverity.WARNING


def test_default_recommendation_strategy_deterministic() -> None:
    severity_strategy = ConservativeReliabilitySeverityStrategy()
    recommendation_strategy = ConservativeReliabilityRecommendationStrategy()
    severity = severity_strategy.classify(_context())
    first = recommendation_strategy.recommend(_context(), severity)
    second = recommendation_strategy.recommend(_context(), severity)
    assert first == second
    assert (
        first.recommendation_kind
        is ExternalEffectReliabilityOperatorRecommendationKind.WAIT_FOR_TRUTH
    )


def test_default_recommendation_has_no_payment_wording() -> None:
    severity = ConservativeReliabilitySeverityStrategy().classify(_context())
    recommendation = ConservativeReliabilityRecommendationStrategy().recommend(
        _context(),
        severity,
    )
    lowered = recommendation.safe_explanation.lower()
    assert "payment" not in lowered
    assert "charge" not in lowered
    assert "customer" not in lowered


def test_custom_severity_plugin_overrides_default() -> None:
    service = build_reliability_diagnostic_classification_service(
        severity_strategy=CriticalTruthUnavailableSeverityStrategy(),
    )
    result = service.classify(_context())
    assert result.severity is not None
    assert result.severity.severity is ExternalEffectReliabilityDiagnosticSeverity.CRITICAL


def test_custom_recommendation_plugin_escalates_on_critical_severity() -> None:
    service = build_reliability_diagnostic_classification_service(
        severity_strategy=CriticalTruthUnavailableSeverityStrategy(),
        recommendation_strategy=EscalateOnCriticalRecommendationStrategy(),
    )
    result = service.classify(_context())
    assert result.recommendation is not None
    assert (
        result.recommendation.recommendation_kind
        is ExternalEffectReliabilityOperatorRecommendationKind.ESCALATE_TO_OPERATOR
    )


def test_same_observation_different_severity_under_different_strategies() -> None:
    default_service = build_reliability_diagnostic_classification_service()
    plugin_service = build_reliability_diagnostic_classification_service(
        severity_strategy=CriticalTruthUnavailableSeverityStrategy(),
    )
    ctx = _context()
    default_result = default_service.classify(ctx)
    plugin_result = plugin_service.classify(ctx)
    assert default_result.severity is not None
    assert plugin_result.severity is not None
    assert default_result.severity.severity is ExternalEffectReliabilityDiagnosticSeverity.WARNING
    assert plugin_result.severity.severity is ExternalEffectReliabilityDiagnosticSeverity.CRITICAL


def test_recommendation_receives_explicit_severity() -> None:
    captured: list[ExternalEffectReliabilitySeverityDecision] = []

    class RecordingRecommendationStrategy(ConservativeReliabilityRecommendationStrategy):
        def recommend(self, context, severity):
            captured.append(severity)
            return super().recommend(context, severity)

    severity_strategy = CriticalTruthUnavailableSeverityStrategy()
    service = build_reliability_diagnostic_classification_service(
        severity_strategy=severity_strategy,
        recommendation_strategy=RecordingRecommendationStrategy(),
    )
    service.classify(_context())
    assert len(captured) == 1
    assert captured[0].severity is ExternalEffectReliabilityDiagnosticSeverity.CRITICAL


def test_invalid_severity_decision_rejected() -> None:
    strategy = ConservativeReliabilitySeverityStrategy()
    bad = ExternalEffectReliabilitySeverityDecision(
        severity=ExternalEffectReliabilityDiagnosticSeverity.ERROR,
        strategy_id=ReliabilityDiagnosticSeverityStrategyId("wrong-id"),
        strategy_version=strategy.strategy_version,
    )
    with pytest.raises(ReliabilityDiagnosticClassificationValidationError):
        validate_severity_decision(bad, strategy)


class _ExplodingSeverityStrategy(ConservativeReliabilitySeverityStrategy):
    def classify(self, context):
        raise RuntimeError("plugin exploded")


def test_severity_strategy_exception_uses_safe_fallback() -> None:
    service = ReliabilityDiagnosticClassificationService(
        severity_strategy=_ExplodingSeverityStrategy(),
        recommendation_strategy=ConservativeReliabilityRecommendationStrategy(),
        severity_fallback=ConservativeReliabilitySeverityStrategy(),
        recommendation_fallback=ConservativeReliabilityRecommendationStrategy(),
    )
    result = service.classify(_context())
    assert result.severity_strategy_failed is True
    assert result.severity is not None
    assert result.severity.severity is ExternalEffectReliabilityDiagnosticSeverity.WARNING


def test_bridge_has_no_severity_mapping_from_signal_kind() -> None:
    mild = map_handoff_to_platform_problem_signal(
        map_observation_to_handoff(
            _observation(ExternalEffectReliabilitySignalKind.UNCERTAINTY_ADMITTED),
        ),
    )
    severe = map_handoff_to_platform_problem_signal(
        map_observation_to_handoff(
            _observation(ExternalEffectReliabilitySignalKind.TRUTH_UNAVAILABLE),
        ),
    )
    assert mild.severity == severe.severity == PROBLEM_SEVERITY_ERROR


def test_mandatory_pluginability_acceptance() -> None:
    ctx = _context(ExternalEffectReliabilitySignalKind.TRUTH_UNAVAILABLE)
    default = build_reliability_diagnostic_classification_service().classify(ctx)
    plugin = build_reliability_diagnostic_classification_service(
        severity_strategy=CriticalTruthUnavailableSeverityStrategy(),
        recommendation_strategy=EscalateOnCriticalRecommendationStrategy(),
    ).classify(ctx)
    assert default.severity is not None and default.recommendation is not None
    assert plugin.severity is not None and plugin.recommendation is not None
    assert default.severity.severity is ExternalEffectReliabilityDiagnosticSeverity.WARNING
    assert plugin.severity.severity is ExternalEffectReliabilityDiagnosticSeverity.CRITICAL
    assert (
        default.recommendation.recommendation_kind
        is ExternalEffectReliabilityOperatorRecommendationKind.WAIT_FOR_TRUTH
    )
    assert (
        plugin.recommendation.recommendation_kind
        is ExternalEffectReliabilityOperatorRecommendationKind.ESCALATE_TO_OPERATOR
    )


def test_invalid_recommendation_output_rejected() -> None:
    class BadRecommendationStrategy(ConservativeReliabilityRecommendationStrategy):
        def recommend(self, context, severity):
            decision = super().recommend(context, severity)
            return decision.__class__(
                recommendation_kind=decision.recommendation_kind,
                strategy_id=decision.strategy_id,
                strategy_version=ReliabilityDiagnosticRecommendationStrategyVersion(""),
                reason_code=decision.reason_code,
                safe_explanation=decision.safe_explanation,
            )

    service = ReliabilityDiagnosticClassificationService(
        severity_strategy=ConservativeReliabilitySeverityStrategy(),
        recommendation_strategy=BadRecommendationStrategy(),
    )
    result = service.classify(_context())
    assert result.recommendation_strategy_failed is True
