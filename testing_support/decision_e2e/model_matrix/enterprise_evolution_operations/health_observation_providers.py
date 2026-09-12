# © Artur Czarnecki. All rights reserved.

"""Evolution health observation provider plugins (DS-E2E-15J-L14)."""

from __future__ import annotations

from dataclasses import dataclass

from testing_support.decision_e2e.model_matrix.enterprise_evolution_operations.contracts import (
    AdaptationOperationalReference,
    EvolutionHealthObservation,
    EvolutionOperationStatus,
)

_DEFAULT_HEALTH_PROVIDER_ID = "default_evolution_health_observation"
_DEFAULT_HEALTH_PROVIDER_VERSION = "1"


@dataclass(frozen=True, slots=True)
class DefaultEvolutionHealthObservationProvider:
    @property
    def provider_id(self) -> str:
        return _DEFAULT_HEALTH_PROVIDER_ID

    @property
    def provider_version(self) -> str:
        return _DEFAULT_HEALTH_PROVIDER_VERSION

    def observe(
        self,
        adaptation_reference: AdaptationOperationalReference,
        *,
        operational_status: EvolutionOperationStatus,
    ) -> EvolutionHealthObservation:
        if operational_status is EvolutionOperationStatus.FAILED:
            quality = "degraded"
            risk = "high"
        elif operational_status is EvolutionOperationStatus.REVIEW_REQUIRED:
            quality = "uncertain"
            risk = "medium"
        elif operational_status is EvolutionOperationStatus.PAUSED:
            quality = "stable"
            risk = "low"
        else:
            quality = "stable"
            risk = "low"
        return EvolutionHealthObservation(
            adaptation_id=adaptation_reference.adaptation_id,
            version=adaptation_reference.version,
            operational_status=operational_status,
            quality_indicator=quality,
            risk_indicator=risk,
            summary=(
                f"Adaptation {adaptation_reference.adaptation_id} "
                f"status={operational_status.value}, "
                f"quality={quality}, risk={risk}."
            ),
        )


def default_evolution_health_observation_provider() -> (
    DefaultEvolutionHealthObservationProvider
):
    return DefaultEvolutionHealthObservationProvider()


__all__ = [
    "DefaultEvolutionHealthObservationProvider",
    "default_evolution_health_observation_provider",
]
