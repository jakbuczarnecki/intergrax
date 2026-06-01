# © Artur Czarnecki. All rights reserved.

"""Assemble lab execution routes + debug inspection API (Phase L.3)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI

from intergrax.debug.app import create_debug_app
from intergrax.llm_adapters.tracking.exposition import register_llm_metrics_routes
from intergrax.debug.hitl_service import DebugHitlResumeService
from intergrax.debug.interaction_service import DebugInteractionIntakeService
from intergrax.runtime.interactions.router import create_interaction_intake_router
from intergrax.runtime.interactions.verification.factory import create_inbound_verifier
from intergrax.runtime.long_running.wiring import wire_long_running_scheduler
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
from intergrax.applications._shared.fastapi_mcp import (
    couple_fastapi_with_mcp,
    make_scheduler_lifespan,
)
from lab_application.host.integration_wiring import wire_lab_integrations
from lab_application.host.settings import LabApplicationSettings
from lab_application.host.tool_wiring import wire_lab_tools
from lab_application.host.wiring import build_lab_registry
from lab_application.mcp.server import build_lab_mcp_server
from intergrax.applications._shared.task_defaults import make_lab_harness_task_enricher
from intergrax.applications._shared.platform_wiring import bootstrap_nexus_platform
from intergrax.applications._shared.task_memory_wiring import wire_task_memory
from intergrax.applications._shared.plugin_bootstrap import attach_plugin_shutdown
from lab_application.serving.fastapi_router import mount_lab_routes


def create_lab_application(
    *,
    settings: Optional[LabApplicationSettings] = None,
    db_path: Path | None = None,
    experiments_db_path: Path | None = None,
    runtime_events_db_path: Path | None = None,
    checkpoints_db_path: Path | None = None,
    registry: Optional[AgentRegistry] = None,
) -> FastAPI:
    """
    Universal Tier-3 lab environment.

    Integrations are composed via ``IntegrationProfile.lab()`` (Phase M.8):
    sqlite persistence, log notifications, lab_json interaction surface (configurable).

    Combines:

    - ``POST /v1/lab/run`` — execute arbitrary registered agents via UnifiedTaskRunner
    - ``GET /v1/lab/agents`` — list active agents and capabilities
    - ``POST /v1/interactions/intake`` — production Slack / Teams / lab inbound webhooks (B.12)
    - Long-running scheduler — HITL timeout / delayed resume (B.05)
    - ``/debug/*`` — trace, events, checkpoints, progress, experiments, HITL intake
    """
    settings = settings or LabApplicationSettings.from_env()
    integrations = wire_lab_integrations(
        settings=settings,
        db_path=db_path,
        experiments_db_path=experiments_db_path,
        runtime_events_db_path=runtime_events_db_path,
        checkpoints_db_path=checkpoints_db_path,
        harness=settings.harness,
        otel_enabled=settings.otel_enabled,
    )
    resolved_registry = registry or build_lab_registry(
        settings=settings,
        integration_profile=integrations.profile,
    )
    task_memory = wire_task_memory(warn_if_disabled=settings.harness)
    nexus_loop = NexusLoop(
        resolved_registry,
        checkpoint_store=integrations.checkpoint_store,
        trace_store=integrations.trace_store,
        runtime_event_store=integrations.runtime_event_store,
        notification_adapter=integrations.notification_adapter,
        task_memory_store=task_memory.store,
        task_memory_db_path=task_memory.db_path,
    )
    plugin_bootstrap = bootstrap_nexus_platform(
        nexus_loop,
        trace_store=integrations.trace_store,  # type: ignore[arg-type]
    )
    task_runner = UnifiedTaskRunner(nexus_loop)
    scheduler_wiring = wire_long_running_scheduler(
        checkpoint_store=integrations.checkpoint_store,
        task_runner=task_runner,
        notification_adapter=integrations.notification_adapter,
        poll_interval_seconds=settings.scheduler_poll_seconds,
        enabled=settings.include_scheduler,
    )
    task_enricher = make_lab_harness_task_enricher(
        default_notify_channel=integrations.default_long_running_notify_channel,
        harness=settings.harness,
    )
    interaction_service = DebugInteractionIntakeService(
        nexus_loop=nexus_loop,
        adapter=integrations.interaction_adapter,
        verifier=create_inbound_verifier(),
        task_enricher=task_enricher,
    )
    hitl_service = DebugHitlResumeService(
        resolved_registry,
        checkpoint_store=integrations.checkpoint_store,
    )

    app = create_debug_app(
        db_path=integrations.trace_db_path,
        experiments_db_path=integrations.experiments_db_path,
        runtime_events_db_path=integrations.runtime_events_db_path,
        checkpoints_db_path=integrations.checkpoints_db_path,
        registry=resolved_registry,
        nexus_loop=nexus_loop,
        interaction_service=interaction_service,
        hitl_service=hitl_service,
        checkpoint_store=integrations.checkpoint_store,
        trace_store=integrations.trace_store,
        runtime_event_store=integrations.runtime_event_store,
        delivery_ledger=integrations.delivery_ledger,
    )
    app.title = "Intergrax Lab Application"
    app.description = (
        "Agent OS experimentation environment — run agents, inspect traces, "
        "checkpoints, runtime events, and experiments (Phase L.3)."
    )
    mount_lab_routes(
        app,
        nexus_loop=nexus_loop,
        prefix=settings.route_prefix,
        task_enricher=task_enricher,
    )
    if settings.include_interaction_routes:
        app.include_router(
            create_interaction_intake_router(
                interaction_service,
                execute_default=True,
            ),
            prefix=settings.interaction_route_prefix,
        )
    scheduler = scheduler_wiring.scheduler if scheduler_wiring is not None else None
    if settings.include_mcp:
        tool_wiring = wire_lab_tools(
            integration_profile=integrations.profile,
            harness=settings.harness,
        )
        mcp = build_lab_mcp_server(
            nexus_loop=nexus_loop,
            route_prefix=settings.route_prefix,
            tool_registry=tool_wiring.registry,
        )
        extra_lifespans = [make_scheduler_lifespan(scheduler)] if scheduler else []
        app = couple_fastapi_with_mcp(
            app,
            mcp,
            mount_path=settings.mcp_mount_path,
            extra_lifespans=extra_lifespans,
        )
    elif scheduler is not None:

        @app.on_event("startup")
        async def _start_long_running_scheduler() -> None:
            await scheduler.start()

        @app.on_event("shutdown")
        async def _stop_long_running_scheduler() -> None:
            await scheduler.stop()
    attach_plugin_shutdown(app, plugin_bootstrap.shutdown_callbacks)
    register_llm_metrics_routes(app)
    return app
