# © Artur Czarnecki. All rights reserved.

"""Shared circuit breaker for integration adapter calls (Phase W-OPS.2)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generic, TypeVar

from intergrax.integrations.contracts.base import IntegrationDependencyError

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"


@dataclass(frozen=True, slots=True)
class IntegrationCircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.recovery_timeout_seconds <= 0:
            raise ValueError("recovery_timeout_seconds must be > 0")


class IntegrationCircuitBreaker:
    """
    In-process circuit breaker for a single integration backend.

    Opens after ``failure_threshold`` consecutive failures; half-open after recovery timeout.
    """

    def __init__(self, name: str, config: IntegrationCircuitBreakerConfig | None = None) -> None:
        self._name = name
        self._config = config or IntegrationCircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at_monotonic: float | None = None

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def name(self) -> str:
        return self._name

    def call(self, operation: Callable[[], T]) -> T:
        self._maybe_transition_to_half_open()
        if self._state == CircuitState.OPEN:
            raise IntegrationDependencyError(
                f"Integration circuit '{self._name}' is open",
                integration_name=self._name,
            )
        try:
            result = operation()
        except IntegrationDependencyError:
            raise
        except Exception as exc:
            self._record_failure()
            raise IntegrationDependencyError(
                f"Integration '{self._name}' call failed: {exc}",
                integration_name=self._name,
            ) from exc
        self._failure_count = 0
        return result

    def _maybe_transition_to_half_open(self) -> None:
        if self._state != CircuitState.OPEN or self._opened_at_monotonic is None:
            return
        elapsed = time.monotonic() - self._opened_at_monotonic
        if elapsed >= self._config.recovery_timeout_seconds:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._opened_at_monotonic = None

    def _record_failure(self) -> None:
        self._failure_count += 1
        if self._failure_count >= self._config.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at_monotonic = time.monotonic()
