# © Artur Czarnecki. All rights reserved.

"""Metrics provider plugins (DS-E2E-15J-L8)."""

from __future__ import annotations

from testing_support.decision_e2e.model_matrix.decision_observability_analytics.contracts import (
    DecisionMetricsSnapshot,
    DecisionObservation,
)


class ObservationVolumeMetricsProvider:
    metrics_provider_id = "observation_volume"
    metrics_provider_version = "1"

    def collect_metrics(
        self,
        observations: tuple[DecisionObservation, ...],
    ) -> DecisionMetricsSnapshot:
        unique_decisions = len({item.decision_id for item in observations})
        return DecisionMetricsSnapshot(
            metrics_provider_id=self.metrics_provider_id,
            metrics_provider_version=self.metrics_provider_version,
            observation_count=len(observations),
            unique_decision_count=unique_decisions,
        )


def default_metrics_providers() -> tuple[ObservationVolumeMetricsProvider, ...]:
    return (ObservationVolumeMetricsProvider(),)


__all__ = [
    "ObservationVolumeMetricsProvider",
    "default_metrics_providers",
]
