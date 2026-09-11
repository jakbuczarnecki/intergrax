# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Statistical forecast analyzer registry (PREDICTIVE R3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from intergrax.contracts.statistical_forecast_analyzer import StatisticalForecastAnalyzer

_T = TypeVar("_T")


class PredictiveForecastRegistryConfigurationError(Exception):
    """Invalid forecast analyzer registration."""


@dataclass(frozen=True, slots=True)
class _OrderedForecastAnalyzer:
    priority: int
    namespace: str
    stable_id: str
    analyzer: StatisticalForecastAnalyzer


def _sort_key(item: _OrderedForecastAnalyzer) -> tuple[int, str, str]:
    return (-item.priority, item.namespace, item.stable_id)


class PredictiveForecastAnalyzerRegistry:
    """Discovery, validation, deterministic ordering for forecast plugins."""

    def __init__(self, analyzers: tuple[StatisticalForecastAnalyzer, ...] = ()) -> None:
        self._analyzers = _order(analyzers)

    @property
    def analyzers(self) -> tuple[StatisticalForecastAnalyzer, ...]:
        return self._analyzers

    @classmethod
    def empty(cls) -> PredictiveForecastAnalyzerRegistry:
        return cls()

    @classmethod
    def platform_default(cls) -> PredictiveForecastAnalyzerRegistry:
        from intergrax.runtime.prediction.forecasting.analyzers.failure_acceleration import (
            FailureAccelerationForecastAnalyzer,
        )
        from intergrax.runtime.prediction.forecasting.analyzers.latency_degradation import (
            LatencyDegradationForecastAnalyzer,
        )
        from intergrax.runtime.prediction.forecasting.analyzers.resource_exhaustion import (
            ResourceExhaustionForecastAnalyzer,
        )
        from intergrax.runtime.prediction.forecasting.analyzers.retry_storm import (
            RetryStormForecastAnalyzer,
        )

        return cls(
            (
                LatencyDegradationForecastAnalyzer(),
                FailureAccelerationForecastAnalyzer(),
                RetryStormForecastAnalyzer(),
                ResourceExhaustionForecastAnalyzer(),
            ),
        )


def _order(analyzers: tuple[StatisticalForecastAnalyzer, ...]) -> tuple[StatisticalForecastAnalyzer, ...]:
    wrapped = [
        _OrderedForecastAnalyzer(
            priority=analyzer.descriptor.priority,
            namespace=analyzer.descriptor.namespace,
            stable_id=analyzer.descriptor.analyzer_id,
            analyzer=analyzer,
        )
        for analyzer in analyzers
    ]
    seen: set[tuple[str, str]] = set()
    for item in wrapped:
        key = (item.namespace, item.stable_id)
        if key in seen:
            raise PredictiveForecastRegistryConfigurationError(
                f"duplicate forecast analyzer: {item.namespace}/{item.stable_id}",
            )
        seen.add(key)
    return tuple(item.analyzer for item in sorted(wrapped, key=_sort_key))


__all__ = [
    "PredictiveForecastAnalyzerRegistry",
    "PredictiveForecastRegistryConfigurationError",
]
