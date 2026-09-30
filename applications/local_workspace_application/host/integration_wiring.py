# © Artur Czarnecki. All rights reserved.

"""Observability wiring for local_workspace_application (product profile)."""

from __future__ import annotations

from pathlib import Path

from intergrax.applications._shared.integration_wiring import bootstrap_application_integration_catalog
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.contracts.host_observability_stores import HostObservabilityStores
from intergrax.runtime.execution.host_observability_composition import wire_host_observability


def wire_local_workspace_integrations(
    *,
    trace_db_path: Path | None = None,
    runtime_events_db_path: Path | None = None,
    integration_profile: IntegrationProfile | None = None,
) -> HostObservabilityStores:
    bootstrap_application_integration_catalog(integration_preset="full")
    return wire_host_observability(
        trace_db_path=trace_db_path,
        runtime_events_db_path=runtime_events_db_path,
        integration_profile=integration_profile or IntegrationProfile(),
    )
