# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI

from intergrax.fastapi_core.app_factory import create_app
from intergrax.fastapi_core.config import ApiConfig, ApiEnvironment
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.applications._shared.fastapi_mcp import couple_fastapi_with_mcp
from intergrax.applications._shared.plugin_bootstrap import attach_plugin_shutdown
from intergrax.applications._shared.platform_wiring import bootstrap_nexus_platform
from research_application.host.integration_wiring import wire_research_integrations
from research_application.host.settings import ResearchBackendSettings
from research_application.host.wiring import build_research_registry
from research_application.host.tool_wiring import wire_research_tools
from research_application.mcp.server import build_research_mcp_server
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

    registry = build_research_registry(settings=settings)
    observability = wire_research_integrations(
        trace_db_path=trace_db_path,
        runtime_events_db_path=runtime_events_db_path,
    )
    nexus = NexusLoop(
        registry,
        trace_store=observability.trace_store,
        runtime_event_store=observability.runtime_event_store,
    )
    platform = bootstrap_nexus_platform(
        nexus,
        trace_store=observability.trace_store,  # type: ignore[arg-type]
    )

    mount_research_routes(
        app,
        nexus_loop=nexus,
        prefix=settings.route_prefix,
    )

    app.title = "Intergrax Research API (prototype)"

    if settings.include_mcp:
        tool_wiring = wire_research_tools(settings=settings)
        mcp = build_research_mcp_server(
            nexus_loop=nexus,
            route_prefix=settings.route_prefix,
            tool_registry=tool_wiring.registry,
        )
        app = couple_fastapi_with_mcp(app, mcp, mount_path=settings.mcp_mount_path)

    attach_plugin_shutdown(app, platform.shutdown_callbacks)
    return app
