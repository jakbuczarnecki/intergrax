# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sanctioned Integrations-owned circuit-breaker composition surface (EBH-3-R2)."""

from __future__ import annotations

from intergrax.integrations._shared.circuit_breaker import IntegrationCircuitBreaker
from intergrax.integrations.contracts.circuit_breaker import (
    IntegrationCircuitBreakerConfig,
    IntegrationCircuitBreakerPort,
)


def create_integration_circuit_breaker(
    name: str,
    config: IntegrationCircuitBreakerConfig | None = None,
) -> IntegrationCircuitBreakerPort:
    """Materialize the canonical integration circuit breaker for cross-domain consumers."""
    return IntegrationCircuitBreaker(name, config)


__all__ = [
    "IntegrationCircuitBreakerConfig",
    "IntegrationCircuitBreakerPort",
    "create_integration_circuit_breaker",
]
