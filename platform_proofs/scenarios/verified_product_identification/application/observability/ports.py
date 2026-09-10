"""Observation sink and timing ports."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from platform_proofs.scenarios.verified_product_identification.application.observability.contracts import (
    ProductIdentificationObservation,
)


class ProductIdentificationObservationSinkMode(StrEnum):
    BEST_EFFORT = "best_effort"
    REQUIRED = "required"


class ObservationSinkError(RuntimeError):
    """Raised when observation persistence fails."""


class ProductIdentificationObservationSink(Protocol):
    def record(self, observation: ProductIdentificationObservation) -> None:
        """Append one immutable observation from the production execution path."""


class MonotonicClockPort(Protocol):
    def monotonic_ns(self) -> int:
        """Return a monotonic timestamp in nanoseconds."""
