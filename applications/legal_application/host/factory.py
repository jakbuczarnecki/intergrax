# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Assemble FastAPI Core + Legal serving into a deployable application."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from legal_application.serving.fastapi_router import mount_legal_agent_routes
from intergrax.fastapi_core.app_factory import create_app
from intergrax.fastapi_core.auth.api_key import ApiKeyConfig
from intergrax.fastapi_core.config import ApiConfig, ApiEnvironment
from intergrax.fastapi_core.runs.default_service import DefaultRunService
from intergrax.fastapi_core.runs.store_memory import InMemoryRunStore
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.registry_projection import MaterializedRegistryProjection
from intergrax.runtime.task.nexus_task_execution_adapter import NexusTaskExecutionAdapter

from intergrax.applications._shared.workspace_cleanup_wiring import (
    apply_factory_lifespans,
    build_factory_lifespans,
)
from intergrax.applications._shared.identity_wiring import wire_application_identity
from intergrax.applications._shared.interaction_wiring import wire_interaction_intake_service
from intergrax.applications._shared.plugin_bootstrap import attach_plugin_shutdown
from intergrax.applications._shared.platform_wiring import bootstrap_nexus_platform
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
from intergrax.runtime.interactions.router import create_interaction_intake_router
from intergrax.runtime.long_running.wiring import wire_long_running_scheduler
from intergrax.applications._shared.host_task_execution_wiring import build_environment_host_task_execution
from legal_application.host.settings import LegalBackendSettings
from legal_application.host.wiring import build_legal_environment_profile, build_legal_manifest


def create_legal_backend_app(
    *,
    registry_projection: MaterializedRegistryProjection,
    settings: Optional[LegalBackendSettings] = None,
    trace_db_path: Path | None = None,
    runtime_events_db_path: Path | None = None,
    document_store: object | None = None,
    key_value_cache: object | None = None,
) -> FastAPI:
    """
    Production host: Intergrax FastAPI Core (health, runs, middleware) + Legal Agent routes.

    Environment variables are read when ``settings`` is omitted (see :mod:`legal_application.host.settings`).

    Uvicorn::

        uvicorn legal_application.host.main:app --host 0.0.0.0 --port 8000
    """
    settings = settings or LegalBackendSettings.from_env()

    api_key_config = ApiKeyConfig(keys=settings.api_keys_map) if settings.api_keys_map else None

    manifest = build_legal_manifest(settings)
    env = manifest.environment or build_legal_environment_profile(settings)
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
    nexus_loop = runtime.nexus_loop
    host_execution = build_environment_host_task_execution(nexus_loop, env)
    registry = runtime.registry
    observability = runtime.observability
    platform = bootstrap_nexus_platform(
        nexus_loop,
        trace_store=observability.trace_store,  # type: ignore[arg-type]
    )
    checkpoint_store = open_default_task_checkpoint_persistence()
    task_enricher = build_reliability_task_enricher(
        env,
        agent_checkpoint_store=runtime.agent_checkpoint_store,
        compensation_queue_store=runtime.compensation_queue_store,
        idempotency_store=runtime.reliability.idempotency_store,
    )
    task_runner = build_task_runner_with_enricher(nexus_loop, task_enricher)
    scheduler_wiring = wire_long_running_scheduler(
        checkpoint_store=checkpoint_store,
        task_runner=task_runner,
        notification_adapter=None,
        poll_interval_seconds=settings.scheduler_poll_seconds,
        enabled=settings.include_scheduler,
    )

    run_store = InMemoryRunStore()
    inline_adapter = NexusTaskExecutionAdapter(task_runner)
    run_service = DefaultRunService(run_store, inline_adapter)
    inline_adapter.bind_run_service(run_service)
    if settings.include_queue_worker:
        queue_dependencies = resolve_host_queue_execution_dependencies(runtime)
        queue_wiring = wire_optional_queue_execution(
            enabled=True,
            registry=registry,
            task_runner=task_runner,
            run_service=run_service,
            app_name="legal_nexus_worker",
            kv_store=queue_dependencies.kv_store,
            causal_evidence_persistence=queue_dependencies.causal_evidence_persistence,
        )
        run_service._execution_adapter = queue_wiring.execution_adapter

    api_cfg = ApiConfig(
        environment=settings.environment,
        api_prefix="/v1",
        cors_allow_origins=settings.cors_allow_origins,
        allowed_hosts=settings.allowed_hosts,
        api_key_config=api_key_config,
        run_store=run_store,
        run_service=run_service,
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

    mount_legal_agent_routes(
        app,
        registry=registry,
        default_agent_id=settings.legal_default_agent_id,
        prefix=settings.route_prefix,
        identity_source=settings.identity_source,
        trace_store=observability.trace_store,
        host_execution=host_execution,
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

    if settings.environment == ApiEnvironment.PROD:
        app.title = "Intergrax Legal API"
    else:
        app.title = "Intergrax Legal API (dev)"

    scheduler = scheduler_wiring.scheduler if scheduler_wiring is not None else None
    if settings.include_mcp:
        from intergrax.applications._shared.mcp_import_guard import load_mcp_coupling

        couple_fastapi_with_mcp = load_mcp_coupling()
        from legal_application.mcp.server import build_legal_mcp_server

        mcp = build_legal_mcp_server(
            host_execution=host_execution,
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

    wire_application_identity(
        app,
        env.identity_profile,
        integration_profile=env.integration_profile,
    )
    attach_plugin_shutdown(app, platform.shutdown_callbacks)
    return app
