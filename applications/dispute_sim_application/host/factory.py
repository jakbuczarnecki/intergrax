# © Artur Czarnecki. All rights reserved.

"""Assemble FastAPI Core + product routes for dispute_sim_application."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from intergrax.applications._shared.fastapi_mcp import couple_fastapi_with_mcp
from intergrax.fastapi_core.app_factory import create_app
from intergrax.fastapi_core.auth.api_key import ApiKeyConfig
from intergrax.fastapi_core.config import ApiConfig
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.platform_wiring import bootstrap_nexus_platform
from intergrax.applications._shared.plugin_bootstrap import attach_plugin_shutdown
from dispute_sim_application.host.settings import DisputeSimBackendSettings
from dispute_sim_application.host.environment_profile import build_dispute_sim_environment_profile
from dispute_sim_application.manifest import build_dispute_sim_manifest
from dispute_sim_application.mcp.server import build_dispute_sim_mcp_server
from dispute_sim_application.serving.fastapi_router import mount_dispute_sim_routes


def create_dispute_sim_backend_app(
    *,
    settings: Optional[DisputeSimBackendSettings] = None,
    trace_db_path: Path | None = None,
    runtime_events_db_path: Path | None = None,
) -> FastAPI:
    settings = settings or DisputeSimBackendSettings.from_env()
    api_key_config = ApiKeyConfig(keys=settings.api_keys_map) if settings.api_keys_map else None

    manifest = build_dispute_sim_manifest()
    env = manifest.environment or build_dispute_sim_environment_profile(settings)
    runtime = build_harness_host_runtime(
        manifest,
        env,
        settings=settings,
        trace_db_path=trace_db_path,
        runtime_events_db_path=runtime_events_db_path,
    )
    nexus_loop = runtime.nexus_loop
    registry = runtime.registry
    platform = bootstrap_nexus_platform(
        nexus_loop,
        trace_store=runtime.observability.trace_store,  # type: ignore[arg-type]
    )

    api_cfg = ApiConfig(
        environment=settings.environment,
        api_prefix="/v1",
        cors_allow_origins=settings.cors_allow_origins,
        allowed_hosts=settings.allowed_hosts,
        api_key_config=api_key_config,
    )
    app = create_app(api_cfg)

    if settings.openapi_enabled_override is True:
        app.docs_url = "/docs"
        app.redoc_url = "/redoc"
        app.openapi_url = "/openapi.json"
    elif settings.openapi_enabled_override is False:
        app.docs_url = None
        app.redoc_url = None
        app.openapi_url = None

    if settings.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=sorted(settings.cors_allow_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    mount_dispute_sim_routes(
        app,
        nexus_loop=nexus_loop,
        prefix=settings.route_prefix,
        default_agent_id=settings.default_agent_id,
    )

    app.title = "Intergrax Dispute Sim API" if settings.environment.value == "prod" else "Intergrax Dispute Sim API (dev)"

    if settings.include_mcp:
        mcp = build_dispute_sim_mcp_server(
            nexus_loop=nexus_loop,
            route_prefix=settings.route_prefix,
            tool_registry=runtime.env_wiring.tool_wiring.registry,
        )
        app = couple_fastapi_with_mcp(app, mcp, mount_path=settings.mcp_mount_path)

    attach_plugin_shutdown(app, platform.shutdown_callbacks)
    return app
