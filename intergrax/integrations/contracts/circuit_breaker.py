# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Public integration circuit-breaker contract (EBH-3-R2)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class IntegrationCircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.recovery_timeout_seconds <= 0:
            raise ValueError("recovery_timeout_seconds must be > 0")


class IntegrationCircuitBreakerPort(Protocol):
    """Structural port for integration-owned in-process circuit breaking."""

    def call(self, operation: Callable[[], T]) -> T: ...
