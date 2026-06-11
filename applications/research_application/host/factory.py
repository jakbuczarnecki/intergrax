# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI

from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.fastapi_core.app_factory import create_app
from intergrax.fastapi_core.config import ApiConfig, ApiEnvironment
from intergrax.applications._shared.fastapi_mcp import (
    apply_lifespans,
    couple_fastapi_with_mcp,
    make_scheduler_lifespan,
)
from intergrax.applications._shared.harness_task_routes import mount_harness_task_routes
from intergrax.applications._shared.identity_wiring import wire_application_identity
from intergrax.applications._shared.interaction_wiring import wire_interaction_intake_service
from intergrax.applications._shared.plugin_bootstrap import attach_plugin_shutdown
from intergrax.applications._shared.platform_wiring import bootstrap_nexus_platform
from intergrax.applications._shared.task_control_wiring import (
    build_reliability_task_enricher,
    build_task_runner_with_enricher,
)
from intergrax.debug.store import open_default_task_checkpoint_persistence
from intergrax.runtime.interactions.router import create_interaction_intake_router
from intergrax.runtime.long_running.wiring import wire_long_running_scheduler
from research_application.host.settings import ResearchBackendSettings
from research_application.host.wiring import build_research_environment_profile, build_research_registry
from research_application.manifest import RESEARCH_APPLICATION_MANIFEST
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

    manifest = RESEARCH_APPLICATION_MANIFEST
    env = manifest.environment or build_research_environment_profile(settings)
    runtime = build_harness_host_runtime(
        manifest.model_copy(update={"environment": env}),
        env,
        settings=settings,
        trace_db_path=trace_db_path,
        runtime_events_db_path=runtime_events_db_path,
    )
    registry = runtime.registry
    nexus = runtime.nexus_loop
    platform = bootstrap_nexus_platform(
        nexus,
        trace_store=runtime.observability.trace_store,  # type: ignore[arg-type]
    )

    checkpoint_store = open_default_task_checkpoint_persistence()
    task_enricher = build_reliability_task_enricher(
        env,
        agent_checkpoint_store=runtime.agent_checkpoint_store,
        compensation_queue_store=runtime.compensation_queue_store,
    )
    task_runner = build_task_runner_with_enricher(nexus, task_enricher)
    scheduler_wiring = wire_long_running_scheduler(
        checkpoint_store=checkpoint_store,
        task_runner=task_runner,
        notification_adapter=None,
        poll_interval_seconds=settings.scheduler_poll_seconds,
        enabled=settings.include_scheduler,
    )

    mount_research_routes(
        app,
        nexus_loop=nexus,
        prefix=settings.route_prefix,
    )

    if settings.include_task_control:
        mount_harness_task_routes(
            app,
            task_runner=task_runner,
            checkpoint_store=checkpoint_store,
            prefix=settings.task_control_route_prefix,
            task_enricher=task_enricher,
        )

    if settings.include_interaction_routes:
        interaction_service = wire_interaction_intake_service(
            nexus,
            interaction_surface=settings.interaction_surface,
            task_enricher=task_enricher,
        )
        app.include_router(
            create_interaction_intake_router(
                interaction_service,
                execute_default=settings.interaction_execute_default,
            ),
            prefix=settings.interaction_route_prefix,
        )

    app.title = "Intergrax Research API (prototype)"

    scheduler = scheduler_wiring.scheduler if scheduler_wiring is not None else None
    if settings.include_mcp:
        mcp = build_research_mcp_server(
            nexus_loop=nexus,
            route_prefix=settings.route_prefix,
            tool_registry=runtime.env_wiring.tool_wiring.registry,
        )
        extra_lifespans = [make_scheduler_lifespan(scheduler)] if scheduler else []
        app = couple_fastapi_with_mcp(
            app,
            mcp,
            mount_path=settings.mcp_mount_path,
            extra_lifespans=extra_lifespans,
        )
    elif scheduler is not None:
        apply_lifespans(app, make_scheduler_lifespan(scheduler))

    wire_application_identity(
        app,
        env.identity_profile,
        integration_profile=env.integration_profile,
    )
    attach_plugin_shutdown(app, platform.shutdown_callbacks)
    return app
