# © Artur Czarnecki. All rights reserved.

"""Assemble lab routes + debug API for poc_template_application."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI

from intergrax.debug.app import create_debug_app
from intergrax.debug.hitl_service import DebugHitlResumeService
from intergrax.debug.interaction_service import DebugInteractionIntakeService
from intergrax.runtime.interactions.router import create_interaction_intake_router
from intergrax.runtime.interactions.verification.factory import create_inbound_verifier
from intergrax.runtime.long_running.wiring import wire_long_running_scheduler
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
from intergrax.applications._shared.platform_wiring import bootstrap_nexus_platform
from intergrax.applications._shared.plugin_bootstrap import attach_plugin_shutdown
from poc_template_application.host.integration_wiring import wire_poc_template_integrations
from poc_template_application.host.settings import PocTemplateApplicationSettings
from poc_template_application.host.tool_wiring import wire_poc_template_tools
from intergrax.applications._shared.fastapi_mcp import (
    couple_fastapi_with_mcp,
    make_scheduler_lifespan,
)
from poc_template_application.host.wiring import build_poc_template_registry
from poc_template_application.mcp.server import build_poc_template_mcp_server
from poc_template_application.serving.fastapi_router import mount_poc_template_routes


def create_poc_template_application(
    *,
    settings: Optional[PocTemplateApplicationSettings] = None,
    db_path: Path | None = None,
    experiments_db_path: Path | None = None,
    runtime_events_db_path: Path | None = None,
    checkpoints_db_path: Path | None = None,
    registry: Optional[AgentRegistry] = None,
) -> FastAPI:
    settings = settings or PocTemplateApplicationSettings.from_env()
    resolved_registry = registry or build_poc_template_registry(settings=settings)
    integrations = wire_poc_template_integrations(
        settings=settings,
        db_path=db_path,
        experiments_db_path=experiments_db_path,
        runtime_events_db_path=runtime_events_db_path,
        checkpoints_db_path=checkpoints_db_path,
    )
    nexus_loop = NexusLoop(
        resolved_registry,
        checkpoint_store=integrations.checkpoint_store,
        trace_store=integrations.trace_store,
        runtime_event_store=integrations.runtime_event_store,
        notification_adapter=integrations.notification_adapter,
    )
    platform = bootstrap_nexus_platform(
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
    interaction_service = DebugInteractionIntakeService(
        nexus_loop=nexus_loop,
        adapter=integrations.interaction_adapter,
        verifier=create_inbound_verifier(),
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
    app.title = "Intergrax Poc Template Lab Application"
    mount_poc_template_routes(app, nexus_loop=nexus_loop, prefix=settings.route_prefix)
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
        tool_wiring = wire_poc_template_tools(
            integration_profile=getattr(integrations, "integration_profile", None),
        )
        mcp = build_poc_template_mcp_server(
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
        async def _start_scheduler() -> None:
            await scheduler.start()

        @app.on_event("shutdown")
        async def _stop_scheduler() -> None:
            await scheduler.stop()
    attach_plugin_shutdown(app, platform.shutdown_callbacks)
    return app
