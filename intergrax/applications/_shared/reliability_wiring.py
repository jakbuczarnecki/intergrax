# © Artur Czarnecki. All rights reserved.

"""Tier-3 reliability wiring (Phase REL-1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.applications._shared.reliability_runtime_bridge import (
    ReliabilityWiringOptions,
    resolve_reliability_wiring_options,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.integrations._shared.circuit_breaker import IntegrationCircuitBreakerConfig
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.runtime.tools.sqlite_idempotency_store import SQLiteIdempotencyStore


@dataclass(frozen=True, slots=True)
class ApplicationReliabilityWiring:
    """Resolved reliability artifacts for a Tier-3 host."""

    options: ReliabilityWiringOptions
    idempotency_store: IdempotencyStore | None
    circuit_breaker_config: IntegrationCircuitBreakerConfig


def wire_application_reliability(
    env: ApplicationEnvironmentProfile,
    *,
    idempotency_db_path: Path | None = None,
) -> ApplicationReliabilityWiring:
    """Materialize idempotency store and circuit breaker config from environment profile."""
    options = resolve_reliability_wiring_options(env.reliability_profile)
    idempotency_store: IdempotencyStore | None = None
    if options.idempotency_enabled:
        if idempotency_db_path is not None:
            idempotency_store = SQLiteIdempotencyStore(str(idempotency_db_path))
        else:
            idempotency_store = InMemoryIdempotencyStore()

    circuit_breaker_config = IntegrationCircuitBreakerConfig(
        failure_threshold=options.circuit_breaker_failure_threshold,
    )
    return ApplicationReliabilityWiring(
        options=options,
        idempotency_store=idempotency_store,
        circuit_breaker_config=circuit_breaker_config,
    )
