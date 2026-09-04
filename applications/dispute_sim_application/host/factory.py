# © Artur Czarnecki. All rights reserved.

"""Assemble FastAPI Core + product routes for dispute_sim_application."""

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
from intergrax.applications._shared.registry_projection import MaterializedRegistryProjection
from intergrax.applications._shared.interaction_wiring import wire_interaction_intake_service
from intergrax.applications._shared.platform_wiring import bootstrap_nexus_platform
from intergrax.applications._shared.plugin_bootstrap import attach_plugin_shutdown
from intergrax.applications._shared.host_queue_execution_wiring import (
    apply_queue_worker_environment_profile,
    resolve_host_queue_execution_dependencies,
)
from intergrax.applications._shared.queue_worker_wiring import wire_optional_queue_execution
from intergrax.applications._shared.task_control_wiring import (
    build_reliability_task_enricher,
    build_task_runner_with_enricher,
    wire_harness_task_control,
)
from intergrax.debug.store import open_default_task_checkpoint_persistence
from intergrax.fastapi_core.runs.default_service import DefaultRunService
from intergrax.fastapi_core.runs.store_memory import InMemoryRunStore
from intergrax.runtime.interactions.router import create_interaction_intake_router
from intergrax.runtime.long_running.wiring import wire_long_running_scheduler
from intergrax.runtime.task.nexus_task_execution_adapter import NexusTaskExecutionAdapter
from intergrax.applications._shared.host_task_execution_wiring import build_environment_host_task_execution
from intergrax.applications._shared.harness_host_runtime_compat import resolve_harness_host_nexus_loop_legacy
from dispute_sim_application.host.settings import DisputeSimBackendSettings
from dispute_sim_application.host.environment_profile import build_dispute_sim_environment_profile
from dispute_sim_application.manifest import build_dispute_sim_manifest
from dispute_sim_application.serving.fastapi_router import mount_dispute_sim_routes


def create_dispute_sim_backend_app(
    *,
    registry_projection: MaterializedRegistryProjection,
    settings: Optional[DisputeSimBackendSettings] = None,
    trace_db_path: Path | None = None,
    runtime_events_db_path: Path | None = None,
    document_store: object | None = None,
    key_value_cache: object | None = None,
) -> FastAPI:
    settings = settings or DisputeSimBackendSettings.from_env()
    api_key_config = ApiKeyConfig(keys=settings.api_keys_map) if settings.api_keys_map else None

    manifest = build_dispute_sim_manifest()
    env = manifest.environment or build_dispute_sim_environment_profile(settings)
    if settings.include_queue_worker and document_store is None and key_value_cache is None:
        env = apply_queue_worker_environment_profile(env)
    runtime = build_harness_host_runtime(
        manifest,
        env,
        settings=settings,
        trace_db_path=trace_db_path,
        runtime_events_db_path=runtime_events_db_path,
        registry_projection=registry_projection,
        document_store=document_store,
        key_value_cache=key_value_cache,
    )
    host_execution = runtime.execution
    nexus_loop = resolve_harness_host_nexus_loop_legacy(runtime)
    platform = bootstrap_nexus_platform(
        nexus_loop,
        trace_store=runtime.observability.trace_store,  # type: ignore[arg-type]
    )

    checkpoint_store = open_default_task_checkpoint_persistence()
    task_enricher = build_reliability_task_enricher(
        env,
        agent_checkpoint_store=runtime.agent_checkpoint_store,
        compensation_queue_store=runtime.compensation_queue_store,
        idempotency_store=runtime.reliability.idempotency_store,
    )
    task_runner = build_task_runner_with_enricher(nexus_loop, task_enricher)
    run_store = InMemoryRunStore()
    inline_adapter = NexusTaskExecutionAdapter(task_runner)
    run_service = DefaultRunService(run_store, inline_adapter)
    inline_adapter.bind_run_service(run_service)
    if settings.include_queue_worker:
        queue_dependencies = resolve_host_queue_execution_dependencies(runtime)
        queue_wiring = wire_optional_queue_execution(
            enabled=True,
            registry=runtime.registry,
            task_runner=task_runner,
            run_service=run_service,
            app_name="dispute_sim_nexus_worker",
            kv_store=queue_dependencies.kv_store,
            causal_evidence_persistence=queue_dependencies.causal_evidence_persistence,
        )
        run_service._execution_adapter = queue_wiring.execution_adapter

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

    mount_dispute_sim_routes(
        app,
        host_execution=host_execution,
        registry=runtime.registry,
        prefix=settings.route_prefix,
        default_agent_id=settings.default_agent_id,
    )

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

    app.title = "Intergrax Dispute Sim API" if settings.environment.value == "prod" else "Intergrax Dispute Sim API (dev)"

    scheduler = scheduler_wiring.scheduler if scheduler_wiring is not None else None
    if settings.include_mcp:
        from intergrax.applications._shared.mcp_import_guard import load_mcp_coupling

        couple_fastapi_with_mcp = load_mcp_coupling()
        from dispute_sim_application.mcp.server import build_dispute_sim_mcp_server

        mcp = build_dispute_sim_mcp_server(
            host_execution=host_execution,
            registry=runtime.registry,
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
