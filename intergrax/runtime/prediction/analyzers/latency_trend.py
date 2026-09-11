# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Rule-based latency trend risk analyzer (PREDICTIVE R1 showcase)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from intergrax.contracts.predictive_analyzer import PredictiveAnalyzer
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
_MODEL_VERSION = "latency_trend@1.0.0"
_LATENCY_INCREASE_RATIO = 0.5
_MIN_EXECUTIONS = 10


@dataclass(frozen=True, slots=True)
class LatencyTrendAnalyzer:
    """Detects degradation risk from rising latency and timeouts — not root cause."""

    analyzer_id: str = "latency_trend"
    analyzer_namespace: str = _PLATFORM_NS
    priority: int = 100
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
    if pattern.avg_latency_ms is None:
        return None

    latency_points = [
        p
        for p in context.performance_history
        if p.metric_name == "latency_ms"
        and (p.component_id is None or p.component_id == pattern.subject_identity)
    ]
    if len(latency_points) < 2:
        return None

    ordered = sorted(latency_points, key=lambda p: p.observed_at)
    baseline = ordered[0].value
    latest = ordered[-1].value
    if baseline <= 0:
        return None
    increase = (latest - baseline) / baseline
    if increase < _LATENCY_INCREASE_RATIO:
        return None

    timeout_ratio = pattern.timeout_count / max(pattern.execution_count, 1)
    severity = PredictiveRiskSeverity.MEDIUM
    confidence = min(0.95, 0.55 + increase * 0.3 + timeout_ratio * 0.2)
    if increase >= 0.8 or timeout_ratio >= 0.15:
        severity = PredictiveRiskSeverity.HIGH
        confidence = min(0.98, confidence + 0.05)

    evidence = (
        f"execution_pattern:{pattern.subject_identity}",
        f"latency_samples:{len(latency_points)}",
    )
    if context.historical_problems:
        evidence += (f"prior_problem:{context.historical_problems[0].problem_id}",)

    return PredictiveRiskSignal(
        signal_id=mint_predictive_signal_id(),
        prediction_run_id=PREDICTION_RUN_ID_ENGINE_STAMP,
        tenant_id=context.tenant_id,
        scope=PredictiveRiskScope.COMPONENT,
        subject_identity=pattern.subject_identity,
        risk_type="degradation_risk",
        severity=severity,
        confidence=confidence,
        evidence_refs=evidence,
        prediction_window=PredictiveWindow(
            duration_seconds=int(timedelta(hours=24).total_seconds()),
            label="next 24h",
        ),
        generated_at=datetime.now(tz=UTC),
        analyzer_metadata=PredictiveAnalyzerMetadata(
            analyzer_id="latency_trend",
            analyzer_version=_MODEL_VERSION,
        ),
        model_version=_MODEL_VERSION,
        summary=(
            f"{pattern.subject_identity} latency trend suggests possible degradation "
            f"(+{int(increase * 100)}% vs baseline window)."
        ),
        recommended_actions=(
            "Review component timeout configuration.",
            "Check downstream dependency availability.",
        ),
    )


__all__ = ["LatencyTrendAnalyzer"]
