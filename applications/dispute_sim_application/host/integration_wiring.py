# © Artur Czarnecki. All rights reserved.

"""Observability wiring for dispute_sim_application (product profile)."""

from __future__ import annotations

from pathlib import Path

from intergrax.applications._shared.integration_wiring import bootstrap_application_integration_catalog
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.nexus.observability_wiring import NexusObservabilityStores, wire_nexus_observability


def wire_dispute_sim_integrations(
    *,
    trace_db_path: Path | None = None,
    runtime_events_db_path: Path | None = None,
    integration_profile: IntegrationProfile | None = None,
) -> NexusObservabilityStores:
    bootstrap_application_integration_catalog(integration_preset="full")
    return wire_nexus_observability(
        trace_db_path=trace_db_path,
        runtime_events_db_path=runtime_events_db_path,
        integration_profile=integration_profile or IntegrationProfile(),
    )
