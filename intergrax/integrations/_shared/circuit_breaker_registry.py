# © Artur Czarnecki. All rights reserved.

"""Per-slug circuit breaker registry for integration calls (Phase W-OPS.2)."""

from __future__ import annotations

from threading import Lock

from intergrax.integrations._shared.circuit_breaker import (
    IntegrationCircuitBreaker,
    IntegrationCircuitBreakerConfig,
)

_registry: dict[str, IntegrationCircuitBreaker] = {}
_lock = Lock()


def get_breaker_for_slug(
    slug: str,
    *,
    config: IntegrationCircuitBreakerConfig | None = None,
) -> IntegrationCircuitBreaker:
    """Return a process-wide breaker for an integration slug."""
    with _lock:
        existing = _registry.get(slug)
        if existing is not None:
            return existing
        breaker = IntegrationCircuitBreaker(slug, config)
        _registry[slug] = breaker
        return breaker


def reset_circuit_breaker_registry_for_tests() -> None:
    """Clear registry between unit tests."""
    with _lock:
        _registry.clear()
