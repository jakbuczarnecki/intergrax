"""In-memory and no-op observation sinks."""

from __future__ import annotations

from dataclasses import dataclass, field

from platform_proofs.scenarios.verified_product_identification.application.observability.contracts import (
    ProductIdentificationObservation,
)
from platform_proofs.scenarios.verified_product_identification.application.observability.ports import (
    ObservationSinkError,
)


@dataclass(frozen=True, slots=True)
class NoOpProductIdentificationObservationSink:
    def record(self, observation: ProductIdentificationObservation) -> None:
        return None


@dataclass
class InMemoryProductIdentificationObservationSink:
    """Collect production-path observations for tests and proof projection."""

    _observations: list[ProductIdentificationObservation] = field(default_factory=list)

    def record(self, observation: ProductIdentificationObservation) -> None:
        self._observations.append(observation)

    def snapshot(self) -> tuple[ProductIdentificationObservation, ...]:
        return tuple(self._observations)


@dataclass(frozen=True, slots=True)
class FailingProductIdentificationObservationSink:
    """Test double that simulates mandatory sink persistence failure."""

    message: str = "sink write failed"

    def record(self, observation: ProductIdentificationObservation) -> None:
        raise ObservationSinkError(self.message)
