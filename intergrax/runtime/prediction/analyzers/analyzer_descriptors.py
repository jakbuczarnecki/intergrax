# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Built-in analyzer governance metadata (PREDICTIVE R2)."""

from __future__ import annotations

from intergrax.contracts.predictive_analyzer_descriptor import AnalyzerDescriptor

LATENCY_TREND_DESCRIPTOR = AnalyzerDescriptor(
    analyzer_id="latency_trend",
    version="1.0",
    owner="intergrax.platform",
    scope="execution",
    required_evidence=("performance_history", "execution_patterns"),
    supported_risk_types=("performance_degradation",),
    limitations=(
        "Rule-based slope only — not root cause.",
        "Requires minimum execution volume.",
    ),
)

FAILURE_PATTERN_DESCRIPTOR = AnalyzerDescriptor(
    analyzer_id="failure_pattern",
    version="1.0",
    owner="intergrax.platform",
    scope="execution",
    required_evidence=("execution_patterns",),
    supported_risk_types=("elevated_failure_rate",),
    limitations=("Heuristic failure-rate threshold — not incident classification.",),
)

CRM_LATENCY_SHOWCASE_DESCRIPTOR = AnalyzerDescriptor(
    analyzer_id="latency-trend",
    version="1.0",
    owner="intergrax.platform",
    scope="execution",
    required_evidence=("latency_ms", "retry_count", "timeout_count"),
    supported_risk_types=("performance_degradation",),
    limitations=("Customer Operations CRM agent scope only in enterprise showcase.",),
)


__all__ = [
    "CRM_LATENCY_SHOWCASE_DESCRIPTOR",
    "FAILURE_PATTERN_DESCRIPTOR",
    "LATENCY_TREND_DESCRIPTOR",
]
