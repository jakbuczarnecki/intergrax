# © Artur Czarnecki. All rights reserved.

"""Tier-0 observability wiring for the research application host."""

from __future__ import annotations

from pathlib import Path

from intergrax.runtime.nexus.observability_wiring import NexusObservabilityStores, wire_nexus_observability


def wire_research_integrations(
    *,
    trace_db_path: Path | None = None,
    runtime_events_db_path: Path | None = None,
) -> NexusObservabilityStores:
    return wire_nexus_observability(
        trace_db_path=trace_db_path,
        runtime_events_db_path=runtime_events_db_path,
    )
