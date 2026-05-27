# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI

from intergrax.fastapi_core.app_factory import create_app
from intergrax.fastapi_core.config import ApiConfig, ApiEnvironment
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore
from research_application.host.settings import ResearchBackendSettings
from research_application.host.wiring import build_research_registry_for_host
from research_application.serving.fastapi_router import mount_research_routes


def create_research_backend_app(
    *,
    settings: Optional[ResearchBackendSettings] = None,
) -> FastAPI:
    settings = settings or ResearchBackendSettings.from_env()
    app = create_app(ApiConfig(environment=ApiEnvironment.DEV))

    registry = build_research_registry_for_host()
    trace_store = InMemoryRunTraceStore()
    nexus = NexusLoop(registry, trace_store=trace_store)

    mount_research_routes(
        app,
        nexus_loop=nexus,
        prefix=settings.route_prefix,
    )

    app.title = "Intergrax Research API (prototype)"
    return app
