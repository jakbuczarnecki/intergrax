# © Artur Czarnecki. All rights reserved.

"""Assemble FastAPI Core + product routes for local_workspace_application."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from intergrax.applications._shared.workspace_cleanup_wiring import (
    apply_factory_lifespans,
    build_factory_lifespans,
)
from intergrax.fastapi_core.app_factory import create_app
from intergrax.fastapi_core.auth.api_key import ApiKeyConfig
from intergrax.fastapi_core.config import ApiConfig
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.harness_task_routes import mount_harness_task_routes
from intergrax.applications._shared.interaction_wiring import wire_interaction_intake_service
from intergrax.applications._shared.platform_wiring import bootstrap_nexus_platform
from intergrax.applications._shared.plugin_bootstrap import attach_plugin_shutdown, bootstrap_application_plugins
from intergrax.runtime.observability.operator_wiring import ObservabilityExportOperatorConfig
from intergrax.applications._shared.task_control_wiring import (
    build_reliability_task_enricher,
    build_task_runner_with_enricher,
)
from intergrax.debug.store import open_default_task_checkpoint_persistence
from intergrax.runtime.interactions.router import create_interaction_intake_router
from intergrax.runtime.long_running.wiring import wire_long_running_scheduler
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.host.environment_profile import build_local_workspace_environment_profile
from local_workspace_application.host.observability_wiring import build_local_workspace_observability_plugins
from local_workspace_application.host.run_task_enricher import build_lkw_http_run_task_enricher
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST
from local_workspace_application.serving.background_task_proof_routes import (
    mount_local_workspace_background_task_proof_routes,
)
from local_workspace_application.serving.fastapi_router import mount_local_workspace_routes
from local_workspace_application.serving.sentry_proof_routes import mount_local_workspace_sentry_proof_routes


def create_local_workspace_backend_app(
    *,
    settings: Optional[LocalWorkspaceBackendSettings] = None,
    trace_db_path: Path | None = None,
    runtime_events_db_path: Path | None = None,
    observability_export: ObservabilityExportOperatorConfig | None = None,
) -> FastAPI:
    settings = settings or LocalWorkspaceBackendSettings.from_env()
    if observability_export is None:
        observability_export = settings.build_observability_export_config()
    api_key_config = ApiKeyConfig(keys=settings.api_keys_map) if settings.api_keys_map else None

    manifest = LOCAL_WORKSPACE_APPLICATION_MANIFEST
    env = manifest.environment or build_local_workspace_environment_profile(settings)
    runtime = build_harness_host_runtime(
        manifest,
        env,
        settings=settings,
        trace_db_path=trace_db_path,
        runtime_events_db_path=runtime_events_db_path,
    )
    nexus_loop = runtime.nexus_loop
    platform = bootstrap_nexus_platform(
        nexus_loop,
        trace_store=runtime.observability.trace_store,  # type: ignore[arg-type]
    )
    lkw_observability_plugins = build_local_workspace_observability_plugins(observability_export)
    if lkw_observability_plugins:
        lkw_plugin_bootstrap = bootstrap_application_plugins(
            list(lkw_observability_plugins),
            nexus_loop=nexus_loop,
        )
        platform.shutdown_callbacks.extend(lkw_plugin_bootstrap.shutdown_callbacks)

    checkpoint_store = open_default_task_checkpoint_persistence()
    task_enricher = build_reliability_task_enricher(
        env,
        agent_checkpoint_store=runtime.agent_checkpoint_store,
        compensation_queue_store=runtime.compensation_queue_store,
        idempotency_store=runtime.reliability.idempotency_store,
    )
    task_runner = build_task_runner_with_enricher(nexus_loop, task_enricher)
    lkw_run_task_enricher = build_lkw_http_run_task_enricher(
        env,
        agent_checkpoint_store=runtime.agent_checkpoint_store,
    )
    lkw_run_task_runner = build_task_runner_with_enricher(nexus_loop, lkw_run_task_enricher)
    scheduler_wiring = wire_long_running_scheduler(
        checkpoint_store=checkpoint_store,
        task_runner=task_runner,
        notification_adapter=None,
        poll_interval_seconds=settings.scheduler_poll_seconds,
        enabled=settings.include_scheduler,
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

    mount_local_workspace_routes(
        app,
        nexus_loop=nexus_loop,
        prefix=settings.route_prefix,
        default_agent_id=settings.default_agent_id,
        task_runner=lkw_run_task_runner,
    )
    mount_local_workspace_sentry_proof_routes(
        app,
        settings=settings,
        observability_export=observability_export,
        prefix=settings.route_prefix,
    )
    mount_local_workspace_background_task_proof_routes(
        app,
        settings=settings,
        wiring_context=runtime.env_wiring.tool_wiring.wiring_context,
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
            nexus_loop,
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

    app.title = "Intergrax Local Workspace API" if settings.environment.value == "prod" else "Intergrax Local Workspace API (dev)"

    scheduler = scheduler_wiring.scheduler if scheduler_wiring is not None else None
    if settings.include_mcp:
        from intergrax.applications._shared.mcp_import_guard import load_mcp_coupling

        couple_fastapi_with_mcp = load_mcp_coupling()
        from local_workspace_application.mcp.server import build_local_workspace_mcp_server

        mcp = build_local_workspace_mcp_server(
            nexus_loop=nexus_loop,
            route_prefix=settings.route_prefix,
            tool_registry=runtime.env_wiring.tool_wiring.registry,
        )
        extra_lifespans = build_factory_lifespans(
            runtime,
            schedulers=[scheduler] if scheduler else None,
        )
        app = couple_fastapi_with_mcp(
            app,
            mcp,
            mount_path=settings.mcp_mount_path,
            extra_lifespans=extra_lifespans,
        )
    else:
        apply_factory_lifespans(app, runtime, schedulers=[scheduler] if scheduler else None)

    attach_plugin_shutdown(app, platform.shutdown_callbacks)
    return app
