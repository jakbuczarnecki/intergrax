# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI

from intergrax.fastapi_core.app_factory import create_app
from intergrax.fastapi_core.config import ApiConfig, ApiEnvironment
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.observability_wiring import wire_nexus_observability
from research_application.host.settings import ResearchBackendSettings
from research_application.host.wiring import build_research_registry_for_host
from research_application.serving.fastapi_router import mount_research_routes


def create_research_backend_app(
    *,
    settings: Optional[ResearchBackendSettings] = None,
    trace_db_path: Path | None = None,
    runtime_events_db_path: Path | None = None,
) -> FastAPI:
    settings = settings or ResearchBackendSettings.from_env()
    if not settings.use_nexus_loop:
        raise ValueError(
            "Research backend requires NexusLoop (§41). "
            "Remove RESEARCH_USE_LEGACY_AGENT_ENGINE or set RESEARCH_USE_NEXUS_LOOP=true."
        )
    app = create_app(ApiConfig(environment=ApiEnvironment.DEV))

    registry = build_research_registry_for_host()
    observability = wire_nexus_observability(
        trace_db_path=trace_db_path,
        runtime_events_db_path=runtime_events_db_path,
    )
    nexus = NexusLoop(
        registry,
        trace_store=observability.trace_store,
        runtime_event_store=observability.runtime_event_store,
    )

    mount_research_routes(
        app,
        nexus_loop=nexus,
        prefix=settings.route_prefix,
    )

    app.title = "Intergrax Research API (prototype)"
    return app
