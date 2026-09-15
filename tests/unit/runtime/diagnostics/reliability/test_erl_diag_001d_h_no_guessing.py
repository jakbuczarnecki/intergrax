# © Artur Czarnecki. All rights reserved.

"""ERL-DIAG-001D-H — fact-bound default classification (no guessing)."""

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
    ExternalEffectReliabilitySignalKind,
    ReliabilityDiagnosticArtifactRefs,
    ReliabilityDiagnosticCorrelation,
    build_reliability_diagnostic_classification_context,
)
from intergrax.runtime.diagnostics.reliability.conservative_reliability_recommendation_strategy import (
    ConservativeReliabilityRecommendationStrategy,
)
from intergrax.runtime.diagnostics.reliability.conservative_reliability_severity_strategy import (
    ConservativeReliabilitySeverityStrategy,
)
from tests.unit.erl_diagnostics_plugins.critical_truth_unavailable_severity_strategy import (
    CriticalTruthUnavailableSeverityStrategy,
)
from tests.unit.erl_diagnostics_plugins.escalate_on_critical_recommendation_strategy import (
    EscalateOnCriticalRecommendationStrategy,
)
from intergrax.runtime.diagnostics.reliability.reliability_diagnostic_strategy_composition import (
    build_reliability_diagnostic_classification_service,
)

pytestmark = pytest.mark.unit

_RECORDED_AT = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)

_ALL_SIGNAL_KINDS = tuple(ExternalEffectReliabilitySignalKind)


def _observation(
    signal_kind: ExternalEffectReliabilitySignalKind,
    *,
    safety: AutomationSafetyHint = AutomationSafetyHint.SAFE,
) -> ExternalEffectReliabilityObservation:
    return ExternalEffectReliabilityObservation(
        observation_id="obs-001d-h",
        tenant_id="tenant-001d-h",
        signal_kind=signal_kind,
        recorded_at=_RECORDED_AT,
        reliability_case_id="case-001d-h",
        correlation=ReliabilityDiagnosticCorrelation(
            tenant_id="tenant-001d-h",
            correlation_id="corr-001d-h",
            reliability_case_id="case-001d-h",
            external_effect_contract_id="contract-1",
        ),
        lifecycle_state=ReliabilityCaseLifecycleState.UNKNOWN_DETECTED,
        artifact_refs=ReliabilityDiagnosticArtifactRefs(evidence_ref="evidence-1"),
        execution_safety_hint=safety,
    )


def _classify(signal_kind: ExternalEffectReliabilitySignalKind):
    ctx = build_reliability_diagnostic_classification_context(_observation(signal_kind))
    severity = ConservativeReliabilitySeverityStrategy().classify(ctx)
    recommendation = ConservativeReliabilityRecommendationStrategy().recommend(ctx, severity)
    return severity, recommendation


def test_truth_established_does_not_imply_no_action_or_success() -> None:
    severity, recommendation = _classify(ExternalEffectReliabilitySignalKind.TRUTH_ESTABLISHED)
    assert severity.severity is not ExternalEffectReliabilityDiagnosticSeverity.INFO
    assert severity.severity is ExternalEffectReliabilityDiagnosticSeverity.WARNING
    assert (
        recommendation.recommendation_kind
        is not ExternalEffectReliabilityOperatorRecommendationKind.NO_OPERATOR_ACTION_REQUIRED
    )
    assert recommendation.recommendation_kind is ExternalEffectReliabilityOperatorRecommendationKind.OBSERVE


def test_governance_posture_does_not_request_approval_by_default() -> None:
    _, recommendation = _classify(ExternalEffectReliabilitySignalKind.GOVERNANCE_POSTURE)
    assert (
        recommendation.recommendation_kind
        is not ExternalEffectReliabilityOperatorRecommendationKind.REQUEST_APPROVAL
    )
    assert recommendation.recommendation_kind is ExternalEffectReliabilityOperatorRecommendationKind.OBSERVE


def test_resolution_posture_neutral_observe_only() -> None:
    severity, recommendation = _classify(ExternalEffectReliabilitySignalKind.RESOLUTION_POSTURE)
    assert severity.severity is ExternalEffectReliabilityDiagnosticSeverity.WARNING
    assert recommendation.recommendation_kind is ExternalEffectReliabilityOperatorRecommendationKind.OBSERVE
    lowered = recommendation.safe_explanation.lower()
    assert "success" not in lowered
    assert "failed" not in lowered


def test_recovery_posture_does_not_imply_recovery_outcome() -> None:
    _, recommendation = _classify(ExternalEffectReliabilitySignalKind.RECOVERY_POSTURE)
    assert recommendation.recommendation_kind is ExternalEffectReliabilityOperatorRecommendationKind.OBSERVE
    lowered = recommendation.safe_explanation.lower()
    assert "succeeded" not in lowered
    assert "continue" not in lowered


def test_truth_unavailable_wait_for_truth() -> None:
    _, recommendation = _classify(ExternalEffectReliabilitySignalKind.TRUTH_UNAVAILABLE)
    assert (
        recommendation.recommendation_kind
        is ExternalEffectReliabilityOperatorRecommendationKind.WAIT_FOR_TRUTH
    )


def test_evidence_insufficient_review_evidence() -> None:
    _, recommendation = _classify(ExternalEffectReliabilitySignalKind.EVIDENCE_INSUFFICIENT)
    assert (
        recommendation.recommendation_kind
        is ExternalEffectReliabilityOperatorRecommendationKind.REVIEW_EVIDENCE
    )


def test_automation_safety_limit_elevated_severity_and_manual_investigation() -> None:
    severity, recommendation = _classify(
        ExternalEffectReliabilitySignalKind.AUTOMATION_SAFETY_LIMIT,
    )
    assert severity.severity is ExternalEffectReliabilityDiagnosticSeverity.ERROR
    assert (
        recommendation.recommendation_kind
        is ExternalEffectReliabilityOperatorRecommendationKind.MANUAL_INVESTIGATION
    )


def test_default_never_emits_request_approval_or_no_action_for_posture_or_truth() -> None:
    forbidden_kinds = (
        ExternalEffectReliabilitySignalKind.TRUTH_ESTABLISHED,
        ExternalEffectReliabilitySignalKind.GOVERNANCE_POSTURE,
        ExternalEffectReliabilitySignalKind.RECOVERY_POSTURE,
        ExternalEffectReliabilitySignalKind.RESOLUTION_POSTURE,
    )
    for signal_kind in forbidden_kinds:
        _, recommendation = _classify(signal_kind)
        assert recommendation.recommendation_kind not in (
            ExternalEffectReliabilityOperatorRecommendationKind.REQUEST_APPROVAL,
            ExternalEffectReliabilityOperatorRecommendationKind.NO_OPERATOR_ACTION_REQUIRED,
        ), signal_kind


def test_custom_plugins_may_still_escalate_and_use_strong_recommendations() -> None:
    ctx = build_reliability_diagnostic_classification_context(
        _observation(ExternalEffectReliabilitySignalKind.TRUTH_UNAVAILABLE),
    )
    result = build_reliability_diagnostic_classification_service(
        severity_strategy=CriticalTruthUnavailableSeverityStrategy(),
        recommendation_strategy=EscalateOnCriticalRecommendationStrategy(),
    ).classify(ctx)
    assert result.severity is not None
    assert result.severity.severity is ExternalEffectReliabilityDiagnosticSeverity.CRITICAL
    assert result.recommendation is not None
    assert (
        result.recommendation.recommendation_kind
        is ExternalEffectReliabilityOperatorRecommendationKind.ESCALATE_TO_OPERATOR
    )


def test_unsafe_automation_elevates_warning_severity_to_error() -> None:
    ctx = build_reliability_diagnostic_classification_context(
        _observation(
            ExternalEffectReliabilitySignalKind.TRUTH_UNAVAILABLE,
            safety=AutomationSafetyHint.UNSAFE,
        ),
    )
    severity = ConservativeReliabilitySeverityStrategy().classify(ctx)
    assert severity.severity is ExternalEffectReliabilityDiagnosticSeverity.ERROR


@pytest.mark.parametrize("signal_kind", _ALL_SIGNAL_KINDS)
def test_default_severity_never_infers_critical_without_facts(
    signal_kind: ExternalEffectReliabilitySignalKind,
) -> None:
    severity, _ = _classify(signal_kind)
    assert severity.severity is not ExternalEffectReliabilityDiagnosticSeverity.CRITICAL
