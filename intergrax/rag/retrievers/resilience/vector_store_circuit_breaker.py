# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Optional in-process circuit breaker for vector-store retrieval calls."""

from __future__ import annotations

from typing import Callable, TypeVar

from intergrax.integrations.contracts.circuit_breaker import (
    IntegrationCircuitBreakerConfig,
    IntegrationCircuitBreakerPort,
)
from intergrax.integrations.registry.circuit_breakers import create_integration_circuit_breaker

T = TypeVar("T")


class RetrieverVectorCircuitBreaker:
    """Guard retriever operations that hit external vector backends."""

    def __init__(
        self,
        *,
        name: str = "rag.vector_store",
        config: IntegrationCircuitBreakerConfig | None = None,
    ) -> None:
        self._breaker: IntegrationCircuitBreakerPort = create_integration_circuit_breaker(
            name,
            config,
        )

    def call(self, operation: Callable[[], T]) -> T:
        return self._breaker.call(operation)
