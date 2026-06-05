# © Artur Czarnecki. All rights reserved.

"""Tier-3 observability wiring (Phase OBS-1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.applications._shared.observability_runtime_bridge import (
    ObservabilityWiringOptions,
    resolve_observability_wiring_options,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.nexus.observability_wiring import NexusObservabilityStores, wire_nexus_observability


@dataclass(frozen=True, slots=True)
class ApplicationObservabilityWiring:
    """Resolved observability stores and options for a Tier-3 host."""

    options: ObservabilityWiringOptions
    stores: NexusObservabilityStores


def wire_application_observability(
    env: ApplicationEnvironmentProfile,
    *,
    trace_db_path: Path | None = None,
    runtime_events_db_path: Path | None = None,
    integration_profile: IntegrationProfile | None = None,
) -> ApplicationObservabilityWiring:
    """Materialize Nexus observability stores from environment profile."""
    options = resolve_observability_wiring_options(env.observability_profile)
    stores = wire_nexus_observability(
        trace_db_path=trace_db_path,
        runtime_events_db_path=runtime_events_db_path,
        use_in_memory_trace=options.use_in_memory_trace,
        enable_runtime_events=options.enable_runtime_events,
        integration_profile=integration_profile or env.integration_profile,
    )
    return ApplicationObservabilityWiring(options=options, stores=stores)
