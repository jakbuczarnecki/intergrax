# © Artur Czarnecki. All rights reserved.

"""Observability assembly validation for Tier-3 hosts (Phase OBS-2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from intergrax.applications._shared.observability_wiring import ApplicationObservabilityWiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore
from intergrax.runtime.nexus.tracing.sqlite_run_trace_store import SQLiteRunTraceStore


@dataclass(frozen=True, slots=True)
class ObservabilityAssemblyValidationResult:
    """Outcome of observability assembly validation."""

    valid: bool
    errors: tuple[str, ...] = ()


class ObservabilityAssemblyError(ValueError):
    """Raised when observability assembly validation fails."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors: tuple[str, ...] = tuple(errors)
        message = "; ".join(self.errors)
        super().__init__(message)


def _integration_has_observability_backend(profile: IntegrationProfile) -> bool:
    return profile.observability_backend is not None


def validate_observability_wiring(
    wiring: ApplicationObservabilityWiring,
    env: ApplicationEnvironmentProfile,
) -> ObservabilityAssemblyValidationResult:
    """Validate observability stores match environment profile requirements."""
    errors: list[str] = []
    profile = env.observability_profile
    stores = wiring.stores

    if profile.trace_sqlite_enabled:
        if isinstance(stores.trace_store, InMemoryRunTraceStore):
            errors.append("trace_sqlite_enabled requires durable trace store, not in-memory")
        if stores.runtime_event_store is None:
            errors.append("trace_sqlite_enabled requires runtime event journal")

    if profile.otel_enabled and not _integration_has_observability_backend(env.integration_profile):
        errors.append("otel_enabled requires IntegrationProfile.observability_backend")

    if not profile.trace_sqlite_enabled and not isinstance(stores.trace_store, InMemoryRunTraceStore):
        errors.append("trace_sqlite_enabled=False requires in-memory trace store")

    return ObservabilityAssemblyValidationResult(valid=not errors, errors=tuple(errors))


def assert_observability_assembly_valid(
    wiring: ApplicationObservabilityWiring,
    env: ApplicationEnvironmentProfile,
) -> None:
    """Raise :class:`ObservabilityAssemblyError` when observability validation fails."""
    result = validate_observability_wiring(wiring, env)
    if not result.valid:
        raise ObservabilityAssemblyError(result.errors)
