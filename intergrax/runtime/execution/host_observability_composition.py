# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Wire observability stores for host composition (Nexus-private implementation)."""

from __future__ import annotations

from pathlib import Path

from intergrax.contracts.host_observability_stores import HostObservabilityStores
from intergrax.contracts.run_trace_store import RunTraceStore
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.nexus.observability_wiring import (
    NexusObservabilityStores,
    wire_nexus_observability,
)


def wire_host_observability(
    *,
    trace_db_path: Path | None = None,
    runtime_events_db_path: Path | None = None,
    trace_store: RunTraceStore | None = None,
    runtime_event_store: RuntimeEventPersistence | None = None,
    use_in_memory_trace: bool = False,
    enable_runtime_events: bool = True,
    integration_profile: IntegrationProfile | None = None,
) -> HostObservabilityStores:
    wired = wire_nexus_observability(
        trace_db_path=trace_db_path,
        runtime_events_db_path=runtime_events_db_path,
        trace_store=trace_store,
        runtime_event_store=runtime_event_store,
        use_in_memory_trace=use_in_memory_trace,
        enable_runtime_events=enable_runtime_events,
        integration_profile=integration_profile,
    )
    return _to_host_stores(wired)


def _to_host_stores(stores: NexusObservabilityStores) -> HostObservabilityStores:
    return HostObservabilityStores(
        trace_store=stores.trace_store,
        runtime_event_store=stores.runtime_event_store,
        trace_db_path=stores.trace_db_path,
        runtime_events_db_path=stores.runtime_events_db_path,
    )


__all__ = ["wire_host_observability"]
