# © Artur Czarnecki. All rights reserved.

"""Assemble FastAPI Core + product routes for governed_contractor_application."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from intergrax.applications._shared.workspace_cleanup_wiring import (
    apply_factory_lifespans,
    build_factory_lifespans,
)
from intergrax.applications._shared.interaction_wiring import wire_interaction_intake_service
from intergrax.fastapi_core.app_factory import create_app
from intergrax.fastapi_core.auth.api_key import ApiKeyConfig
from intergrax.fastapi_core.config import ApiConfig
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.registry_projection import MaterializedRegistryProjection
from intergrax.applications._shared.platform_wiring import bootstrap_nexus_platform
from intergrax.applications._shared.plugin_bootstrap import (
    attach_plugin_shutdown,
    bootstrap_application_plugins,
)
from intergrax.runtime.observability.operator_wiring import (
    ObservabilityExportOperatorConfig,
    build_observability_export_runtime_plugin,
)
from intergrax.runtime.interactions.router import create_interaction_intake_router
from intergrax.applications._shared.task_control_wiring import (
    build_reliability_task_enricher,
    build_task_runner_with_enricher,
    wire_harness_task_control,
)
from intergrax.applications._shared.product_observability_dashboard_wiring import (
    wire_harness_product_observability_dashboard,
)
from intergrax.debug.store import open_default_task_checkpoint_persistence
from intergrax.runtime.long_running.wiring import wire_long_running_scheduler
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from governed_contractor_application.host.environment_profile import build_governed_contractor_environment_profile
from governed_contractor_application.manifest import build_governed_contractor_manifest
from governed_contractor_application.serving.fastapi_router import mount_governed_contractor_routes


def create_governed_contractor_backend_app(
    *,
    registry_projection: MaterializedRegistryProjection,
    settings: Optional[GovernedContractorBackendSettings] = None,
    trace_db_path: Path | None = None,
    runtime_events_db_path: Path | None = None,
    checkpoints_db_path: Path | None = None,
    document_store: object | None = None,
    observability_export: ObservabilityExportOperatorConfig | None = None,
) -> FastAPI:
    settings = settings or GovernedContractorBackendSettings.from_env()
    api_key_config = ApiKeyConfig(keys=settings.api_keys_map) if settings.api_keys_map else None

    manifest = build_governed_contractor_manifest()
    env = manifest.environment or build_governed_contractor_environment_profile(settings)
    runtime = build_harness_host_runtime(
        manifest,
        env,
        settings=settings,
        trace_db_path=trace_db_path,
        runtime_events_db_path=runtime_events_db_path,
        checkpoints_db_path=checkpoints_db_path,
        registry_projection=registry_projection,
        document_store=document_store,
    )
    nexus_loop = runtime.nexus_loop
    registry = runtime.registry
    platform = bootstrap_nexus_platform(
        nexus_loop,
        trace_store=runtime.observability.trace_store,  # type: ignore[arg-type]
    )
    if observability_export is not None and observability_export.enabled:
        export_plugin = build_observability_export_runtime_plugin(observability_export)
        if export_plugin is not None:
            export_bootstrap = bootstrap_application_plugins(
                [export_plugin],
                nexus_loop=nexus_loop,
            )
            platform.shutdown_callbacks.extend(export_bootstrap.shutdown_callbacks)
    checkpoint_store = open_default_task_checkpoint_persistence(db_path=checkpoints_db_path)
    task_enricher = build_reliability_task_enricher(env)
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

    mount_governed_contractor_routes(
        app,
        nexus_loop=nexus_loop,
        prefix=settings.route_prefix,
        default_agent_id=settings.default_agent_id,
    )

    wire_harness_product_observability_dashboard(app, runtime=runtime)

    if settings.include_task_control:
        wire_harness_task_control(
            app,
            enabled=True,
            task_runner=task_runner,
            env=env,
            checkpoint_store=checkpoint_store,
            task_route_prefix=settings.task_control_route_prefix,
            task_enricher=task_enricher,
            runtime=runtime,
        )

    if settings.include_interaction_routes:
        app.include_router(
            create_interaction_intake_router(
                interaction_service,
                execute_default=settings.interaction_execute_default,
            ),
            prefix=settings.interaction_route_prefix,
        )

    app.title = "Intergrax Governed Contractor API" if settings.environment.value == "prod" else "Intergrax Governed Contractor API (dev)"

    scheduler = scheduler_wiring.scheduler if scheduler_wiring is not None else None
    if settings.include_mcp:
        from intergrax.applications._shared.mcp_import_guard import load_mcp_coupling

        couple_fastapi_with_mcp = load_mcp_coupling()
        from governed_contractor_application.mcp.server import build_governed_contractor_mcp_server

        mcp = build_governed_contractor_mcp_server(
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
    app.state.harness_runtime = runtime
    return app
