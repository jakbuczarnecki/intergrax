# © Artur Czarnecki. All rights reserved.

"""Integration health probes at Tier-3 bootstrap (Phase INT-2)."""

from __future__ import annotations

from intergrax.integrations._shared.circuit_breaker import IntegrationCircuitBreakerConfig
from intergrax.integrations._shared.health import health_check_all
from intergrax.integrations.contracts.base import HealthStatus
from intergrax.integrations.registry.profile import IntegrationProfile


def probe_integration_profile_health(
    profile: IntegrationProfile,
    *,
    use_circuit_breaker: bool = True,
    circuit_breaker_config: IntegrationCircuitBreakerConfig | None = None,
) -> tuple[HealthStatus, ...]:
    """
    Run catalog health probes for all slugs selected by ``profile``.

    Returns an immutable tuple suitable for ``ApplicationEnvironmentWiring``.
    """
    results = health_check_all(
        profile,
        use_circuit_breaker=use_circuit_breaker,
        circuit_breaker_config=circuit_breaker_config,
    )
    return tuple(results)


def integration_health_summary(health: tuple[HealthStatus, ...]) -> str:
    """Compact summary for logs and bootstrap diagnostics."""
    if not health:
        return "no integrations configured"
    healthy = sum(1 for item in health if item.healthy)
    return f"{healthy}/{len(health)} integrations healthy"
