# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Runtime adapter: reliability facts → pluginable severity and recommendation strategies."""

from __future__ import annotations

import logging

from intergrax.contracts.enterprise_reliability.diagnostics.classification import (
    MAX_RELIABILITY_CLASSIFICATION_EXPLANATION_LEN,
    ExternalEffectReliabilityDiagnosticClassificationContext,
    ExternalEffectReliabilityDiagnosticClassificationResult,
    ExternalEffectReliabilityDiagnosticRecommendationStrategy,
    ExternalEffectReliabilityDiagnosticSeverity,
    ExternalEffectReliabilityDiagnosticSeverityStrategy,
    ExternalEffectReliabilityOperatorRecommendationKind,
    ExternalEffectReliabilityRecommendationDecision,
    ExternalEffectReliabilitySeverityDecision,
    ReliabilityDiagnosticRecommendationStrategyId,
    ReliabilityDiagnosticRecommendationStrategyVersion,
    ReliabilityDiagnosticSeverityStrategyId,
    ReliabilityDiagnosticSeverityStrategyVersion,
)

_LOGGER = logging.getLogger(__name__)


class ReliabilityDiagnosticClassificationValidationError(ValueError):
    """Strategy output failed validation."""


class ReliabilityDiagnosticClassificationService:
    """
    Invokes severity and recommendation strategies with validation and safe containment.

    Does not persist Problems, mutate execution, or invoke governance/recovery.
    """

    def __init__(
        self,
        *,
        severity_strategy: ExternalEffectReliabilityDiagnosticSeverityStrategy,
        recommendation_strategy: ExternalEffectReliabilityDiagnosticRecommendationStrategy,
        severity_fallback: ExternalEffectReliabilityDiagnosticSeverityStrategy | None = None,
        recommendation_fallback: ExternalEffectReliabilityDiagnosticRecommendationStrategy | None = None,
    ) -> None:
        self._severity_strategy = severity_strategy
        self._recommendation_strategy = recommendation_strategy
        self._severity_fallback = severity_fallback
        self._recommendation_fallback = recommendation_fallback

    def classify(
        self,
        context: ExternalEffectReliabilityDiagnosticClassificationContext,
    ) -> ExternalEffectReliabilityDiagnosticClassificationResult:
        if type(context) is not ExternalEffectReliabilityDiagnosticClassificationContext:
            raise TypeError("context must be ExternalEffectReliabilityDiagnosticClassificationContext")

        severity_failed = False
        recommendation_failed = False
        severity_decision: ExternalEffectReliabilitySeverityDecision | None = None
        recommendation_decision: ExternalEffectReliabilityRecommendationDecision | None = None

        severity_decision, severity_failed = self._run_severity(context)
        if severity_decision is not None:
            recommendation_decision, recommendation_failed = self._run_recommendation(
                context,
                severity_decision,
            )

        return ExternalEffectReliabilityDiagnosticClassificationResult(
            severity=severity_decision,
            recommendation=recommendation_decision,
            severity_strategy_failed=severity_failed,
            recommendation_strategy_failed=recommendation_failed,
        )

    def _run_severity(
        self,
        context: ExternalEffectReliabilityDiagnosticClassificationContext,
    ) -> tuple[ExternalEffectReliabilitySeverityDecision | None, bool]:
        try:
            decision = self._severity_strategy.classify(context)
            validate_severity_decision(decision, self._severity_strategy)
            return decision, False
        except Exception as exc:
            _LOGGER.warning(
                "erl_reliability_severity_strategy_failed",
                extra={"strategy_id": str(self._severity_strategy.strategy_id)},
                exc_info=exc,
            )
            if self._severity_fallback is not None:
                try:
                    fallback = self._severity_fallback.classify(context)
                    validate_severity_decision(fallback, self._severity_fallback)
                    return fallback, True
                except Exception:
                    _LOGGER.warning(
                        "erl_reliability_severity_fallback_failed",
                        exc_info=True,
                    )
            return None, True

    def _run_recommendation(
        self,
        context: ExternalEffectReliabilityDiagnosticClassificationContext,
        severity: ExternalEffectReliabilitySeverityDecision,
    ) -> tuple[ExternalEffectReliabilityRecommendationDecision | None, bool]:
        try:
            decision = self._recommendation_strategy.recommend(context, severity)
            validate_recommendation_decision(decision, self._recommendation_strategy)
            return decision, False
        except Exception as exc:
            _LOGGER.warning(
                "erl_reliability_recommendation_strategy_failed",
                extra={"strategy_id": str(self._recommendation_strategy.strategy_id)},
                exc_info=exc,
            )
            if self._recommendation_fallback is not None:
                try:
                    fallback = self._recommendation_fallback.recommend(context, severity)
                    validate_recommendation_decision(fallback, self._recommendation_fallback)
                    return fallback, True
                except Exception:
                    _LOGGER.warning(
                        "erl_reliability_recommendation_fallback_failed",
                        exc_info=True,
                    )
            return None, True


def validate_severity_decision(
    decision: ExternalEffectReliabilitySeverityDecision,
    strategy: ExternalEffectReliabilityDiagnosticSeverityStrategy,
) -> None:
    if type(decision) is not ExternalEffectReliabilitySeverityDecision:
        raise ReliabilityDiagnosticClassificationValidationError(
            "severity decision has unexpected type",
        )
    _require_strategy_id(decision.strategy_id, field_name="severity.strategy_id")
    _require_strategy_version(decision.strategy_version, field_name="severity.strategy_version")
    if decision.strategy_id != strategy.strategy_id:
        raise ReliabilityDiagnosticClassificationValidationError(
            "severity decision strategy_id must match invoking strategy",
        )
    if decision.strategy_version != strategy.strategy_version:
        raise ReliabilityDiagnosticClassificationValidationError(
            "severity decision strategy_version must match invoking strategy",
        )
    if decision.severity not in ExternalEffectReliabilityDiagnosticSeverity:
        raise ReliabilityDiagnosticClassificationValidationError("invalid severity enum value")
    if decision.reason_code is not None:
        _require_reason_code(decision.reason_code)
    if decision.safe_explanation is not None:
        _require_bounded_explanation(decision.safe_explanation)


def validate_recommendation_decision(
    decision: ExternalEffectReliabilityRecommendationDecision,
    strategy: ExternalEffectReliabilityDiagnosticRecommendationStrategy,
) -> None:
    if type(decision) is not ExternalEffectReliabilityRecommendationDecision:
        raise ReliabilityDiagnosticClassificationValidationError(
            "recommendation decision has unexpected type",
        )
    _require_strategy_id(decision.strategy_id, field_name="recommendation.strategy_id")
    _require_strategy_version(
        decision.strategy_version,
        field_name="recommendation.strategy_version",
    )
    if decision.strategy_id != strategy.strategy_id:
        raise ReliabilityDiagnosticClassificationValidationError(
            "recommendation decision strategy_id must match invoking strategy",
        )
    if decision.strategy_version != strategy.strategy_version:
        raise ReliabilityDiagnosticClassificationValidationError(
            "recommendation decision strategy_version must match invoking strategy",
        )
    if decision.recommendation_kind not in ExternalEffectReliabilityOperatorRecommendationKind:
        raise ReliabilityDiagnosticClassificationValidationError(
            "invalid recommendation_kind enum value",
        )
    _require_reason_code(decision.reason_code)
    _require_bounded_explanation(decision.safe_explanation)
    for index, ref in enumerate(decision.evidence_refs):
        if not ref or not ref.strip():
            raise ReliabilityDiagnosticClassificationValidationError(
                f"evidence_refs[{index}] must be non-empty",
            )


def _require_strategy_id(
    value: ReliabilityDiagnosticSeverityStrategyId | ReliabilityDiagnosticRecommendationStrategyId,
    *,
    field_name: str,
) -> None:
    if type(value) is not str:
        raise ReliabilityDiagnosticClassificationValidationError(f"{field_name} must be str")
    if not value.strip():
        raise ReliabilityDiagnosticClassificationValidationError(f"{field_name} must be non-empty")


def _require_strategy_version(
    value: ReliabilityDiagnosticSeverityStrategyVersion
    | ReliabilityDiagnosticRecommendationStrategyVersion,
    *,
    field_name: str,
) -> None:
    if type(value) is not str:
        raise ReliabilityDiagnosticClassificationValidationError(f"{field_name} must be str")
    if not value.strip():
        raise ReliabilityDiagnosticClassificationValidationError(f"{field_name} must be non-empty")


def _require_reason_code(value: str) -> None:
    if type(value) is not str or not value.strip():
        raise ReliabilityDiagnosticClassificationValidationError("reason_code must be non-empty str")
    if len(value) > 128:
        raise ReliabilityDiagnosticClassificationValidationError("reason_code exceeds max length")


def _require_bounded_explanation(value: str) -> None:
    if type(value) is not str or not value.strip():
        raise ReliabilityDiagnosticClassificationValidationError(
            "safe_explanation must be non-empty str",
        )
    if len(value) > MAX_RELIABILITY_CLASSIFICATION_EXPLANATION_LEN:
        raise ReliabilityDiagnosticClassificationValidationError(
            "safe_explanation exceeds max length",
        )


__all__ = [
    "ReliabilityDiagnosticClassificationService",
    "ReliabilityDiagnosticClassificationValidationError",
    "validate_recommendation_decision",
    "validate_severity_decision",
]
