# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Rule-based failure-rate pattern analyzer (PREDICTIVE R1)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from intergrax.contracts.predictive_context import (
    ExecutionPatternSnapshot,
    PredictiveContext,
)
from intergrax.contracts.predictive_risk import (
    PREDICTION_RUN_ID_ENGINE_STAMP,
    PredictiveAnalyzerMetadata,
    PredictiveRiskScope,
    PredictiveRiskSeverity,
    PredictiveRiskSignal,
    PredictiveWindow,
    mint_predictive_signal_id,
)

_PLATFORM_NS = "intergrax.platform"
_MODEL_VERSION = "failure_pattern@1.0.0"
_FAILURE_RATE_THRESHOLD = 0.2
_MIN_EXECUTIONS = 5


@dataclass(frozen=True, slots=True)
class FailurePatternAnalyzer:
    """Detects elevated failure rates — probability language only."""

    analyzer_id: str = "failure_pattern"
    analyzer_namespace: str = _PLATFORM_NS
    priority: int = 90
    model_version: str = _MODEL_VERSION

    def analyze(self, context: PredictiveContext) -> tuple[PredictiveRiskSignal, ...]:
        signals: list[PredictiveRiskSignal] = []
        for pattern in context.execution_patterns:
            signal = _signal_for_pattern(context, pattern)
            if signal is not None:
                signals.append(signal)
        return tuple(signals)


def _signal_for_pattern(
    context: PredictiveContext,
    pattern: ExecutionPatternSnapshot,
) -> PredictiveRiskSignal | None:
    if pattern.execution_count < _MIN_EXECUTIONS:
        return None
    rate = pattern.failed_execution_count / pattern.execution_count
    if rate < _FAILURE_RATE_THRESHOLD:
        return None

    severity = PredictiveRiskSeverity.MEDIUM
    confidence = min(0.92, 0.45 + rate)
    if rate >= 0.35:
        severity = PredictiveRiskSeverity.HIGH

    evidence = (
        f"executions:{pattern.execution_count}",
        f"failed:{pattern.failed_execution_count}",
        f"subject:{pattern.subject_identity}",
    )

    return PredictiveRiskSignal(
        signal_id=mint_predictive_signal_id(),
        prediction_run_id=PREDICTION_RUN_ID_ENGINE_STAMP,
        tenant_id=context.tenant_id,
        scope=PredictiveRiskScope.COMPONENT,
        subject_identity=pattern.subject_identity,
        risk_type="failure_probability",
        severity=severity,
        confidence=confidence,
        evidence_refs=evidence,
        prediction_window=PredictiveWindow(
            duration_seconds=int(timedelta(minutes=30).total_seconds()),
            label="next 30 minutes",
        ),
        generated_at=datetime.now(tz=UTC),
        analyzer_metadata=PredictiveAnalyzerMetadata(
            analyzer_id="failure_pattern",
            analyzer_version=_MODEL_VERSION,
        ),
        model_version=_MODEL_VERSION,
        summary=(
            f"{pattern.subject_identity} failure rate ({int(rate * 100)}%) "
            "suggests elevated incident risk."
        ),
        recommended_actions=("Review recent execution failures for this component.",),
    )


__all__ = ["FailurePatternAnalyzer"]
