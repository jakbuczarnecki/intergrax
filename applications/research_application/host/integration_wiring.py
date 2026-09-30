# © Artur Czarnecki. All rights reserved.

"""Tier-0 observability wiring for the research application host."""

from __future__ import annotations

from pathlib import Path

from intergrax.contracts.host_observability_stores import HostObservabilityStores
from intergrax.runtime.execution.host_observability_composition import wire_host_observability


def wire_research_integrations(
    *,
    trace_db_path: Path | None = None,
    runtime_events_db_path: Path | None = None,
) -> HostObservabilityStores:
    return wire_host_observability(
        trace_db_path=trace_db_path,
        runtime_events_db_path=runtime_events_db_path,
    )
