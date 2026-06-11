# © Artur Czarnecki. All rights reserved.

"""Assemble lab routes + debug API for poc_template_application (Phase DX-1.3)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI

from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.fastapi_mcp import (
    apply_lifespans,
    couple_fastapi_with_mcp,
    make_scheduler_lifespan,
)
from intergrax.applications._shared.harness_task_routes import mount_harness_task_routes
from intergrax.applications._shared.interaction_wiring import wire_interaction_intake_service
from intergrax.applications._shared.platform_wiring import bootstrap_nexus_platform
from intergrax.applications._shared.plugin_bootstrap import attach_plugin_shutdown
from intergrax.applications._shared.task_control_wiring import (
    build_reliability_task_enricher,
    build_task_runner_with_enricher,
)
from intergrax.debug.app import create_debug_app
from intergrax.debug.hitl_service import DebugHitlResumeService
from intergrax.debug.store import open_default_task_checkpoint_persistence
from intergrax.runtime.interactions.router import create_interaction_intake_router
from intergrax.runtime.long_running.wiring import wire_long_running_scheduler
from intergrax.runtime.registry.agent_registry import AgentRegistry
from poc_template_application.host.settings import PocTemplateApplicationSettings
from poc_template_application.host.wiring import build_poc_template_registry
from poc_template_application.manifest import build_poc_template_manifest
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
    if registry is None:
        registry = build_poc_template_registry(settings=settings)
    manifest = build_poc_template_manifest()
    env = manifest.resolved_environment()
    runtime = build_harness_host_runtime(
        manifest.model_copy(update={"environment": env}),
        env,
        settings=settings,
        trace_db_path=db_path,
        runtime_events_db_path=runtime_events_db_path,
        use_in_memory_trace=db_path is None,
        checkpoints_db_path=checkpoints_db_path,
    )
    nexus_loop = runtime.nexus_loop
    resolved_registry = registry
    platform = bootstrap_nexus_platform(
        nexus_loop,
        trace_store=runtime.observability.trace_store,  # type: ignore[arg-type]
    )
    checkpoint_store = open_default_task_checkpoint_persistence(db_path=checkpoints_db_path)
    task_enricher = build_reliability_task_enricher(
        env,
        agent_checkpoint_store=runtime.agent_checkpoint_store,
        compensation_queue_store=runtime.compensation_queue_store,
    )
    task_runner = build_task_runner_with_enricher(nexus_loop, task_enricher)
    scheduler_wiring = wire_long_running_scheduler(
        checkpoint_store=checkpoint_store,
        task_runner=task_runner,
        notification_adapter=None,
        poll_interval_seconds=settings.scheduler_poll_seconds,
        enabled=settings.include_scheduler,
    )
    interaction_service = wire_interaction_intake_service(
        nexus_loop,
        interaction_surface=settings.interaction_surface,
        task_enricher=task_enricher,
    )
    hitl_service = DebugHitlResumeService(
        resolved_registry,
        checkpoint_store=checkpoint_store,
    )
    app = create_debug_app(
        db_path=runtime.observability.trace_db_path,
        experiments_db_path=experiments_db_path,
        runtime_events_db_path=runtime.observability.runtime_events_db_path,
        checkpoints_db_path=checkpoints_db_path,
        registry=resolved_registry,
        nexus_loop=nexus_loop,
        interaction_service=interaction_service,
        hitl_service=hitl_service,
        checkpoint_store=checkpoint_store,
        trace_store=runtime.observability.trace_store,
        runtime_event_store=runtime.observability.runtime_event_store,
    )
    app.title = "Intergrax Poc Template Lab Application"
    mount_poc_template_routes(app, nexus_loop=nexus_loop, prefix=settings.route_prefix)
    if settings.include_task_control:
        mount_harness_task_routes(
            app,
            task_runner=task_runner,
            checkpoint_store=checkpoint_store,
            prefix=settings.task_control_route_prefix,
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
        mcp = build_poc_template_mcp_server(
            nexus_loop=nexus_loop,
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
    attach_plugin_shutdown(app, platform.shutdown_callbacks)
    return app
