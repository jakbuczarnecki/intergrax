# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Optional in-process circuit breaker for vector-store retrieval calls."""

from __future__ import annotations

from typing import Callable, Optional, TypeVar

from intergrax.integrations._shared.circuit_breaker import (
    IntegrationCircuitBreaker,
    IntegrationCircuitBreakerConfig,
)

T = TypeVar("T")


class RetrieverVectorCircuitBreaker:
    """Guard retriever operations that hit external vector backends."""

    def __init__(
        self,
        *,
        name: str = "rag.vector_store",
        config: Optional[IntegrationCircuitBreakerConfig] = None,
    ) -> None:
        self._breaker = IntegrationCircuitBreaker(name, config)

    def call(self, operation: Callable[[], T]) -> T:
        return self._breaker.call(operation)
