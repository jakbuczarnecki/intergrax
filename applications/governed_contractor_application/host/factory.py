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
from intergrax.applications._shared.harness_host_auxiliary_wiring import HostTaskExecutionExecutor
from intergrax.applications._shared.interaction_wiring import wire_interaction_intake_service
from intergrax.fastapi_core.app_factory import create_app
from intergrax.fastapi_core.auth.api_key import ApiKeyConfig
from intergrax.fastapi_core.config import ApiConfig
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.production_platform_persistence import (
    resolve_harness_host_profile_persistence_kwargs_from_composition,
    resolve_reference_production_strict_host_environment,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
)
from intergrax.applications._shared.registry_projection import MaterializedRegistryProjection
from intergrax.applications._shared.plugin_bootstrap import (
    attach_plugin_shutdown,
)
from intergrax.runtime.observability.operator_wiring import (
    ObservabilityExportOperatorConfig,
    build_observability_export_runtime_plugin,
)
from intergrax.runtime.interactions.router import create_interaction_intake_router
from intergrax.applications._shared.harness_host_auxiliary_wiring import (
    wire_harness_host_long_running_scheduler,
)
from intergrax.applications._shared.task_control_wiring import (
    build_reliability_task_enricher,
    wire_harness_task_control,
)
from intergrax.applications._shared.harness_host_composition import (
    bootstrap_harness_host_application_plugins,
    bootstrap_harness_host_platform,
    resolve_harness_host_event_bus,
    resolve_harness_host_lifecycle_hook_coordinator,
    resolve_harness_host_middleware_pipeline,
    resolve_harness_host_runtime_event_persistence,
)
from intergrax.applications._shared.product_observability_dashboard_wiring import (
    wire_harness_product_observability_dashboard,
)
from intergrax.debug.store import open_default_task_checkpoint_persistence
from intergrax.applications._shared.harness_host_orchestration_topology_wiring import (
    HarnessHostOrchestrationTopologyReliabilityCompositionError,
)
from governed_contractor_application.host.orchestration_topology_production_composition import (
    build_governed_contractor_production_orchestration_topology_submission_port,
)
from governed_contractor_application.host.governed_contractor_host_runtime_composition import (
    GovernedContractorHostRuntimeComposition,
)
from governed_contractor_application.host.orchestration_decision_requirement_policy import (
    default_governed_contractor_harness_orchestration_decision_requirement_policy,
)
from governed_contractor_application.host.production_external_work_composition import (
    resolve_production_runtime_policy_bundle_evaluator,
)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from governed_contractor_application.host.collaborative_work_integration_profile import (
    resolve_governed_contractor_collaborative_work_integration_profile,
)
from governed_contractor_application.host.environment_profile import build_governed_contractor_environment_profile
from governed_contractor_application.manifest import build_governed_contractor_manifest
from governed_contractor_application.serving.fastapi_router import mount_governed_contractor_routes


def create_governed_contractor_backend_app(
    *,
    registry_projection: MaterializedRegistryProjection,
    process_composition: ProductionProcessComposition | None = None,
    settings: Optional[GovernedContractorBackendSettings] = None,
    host_runtime: GovernedContractorHostRuntimeComposition | None = None,
    trace_db_path: Path | None = None,
    runtime_events_db_path: Path | None = None,
    checkpoints_db_path: Path | None = None,
    document_store: object | None = None,
    key_value_cache: object | None = None,
    execution_continuation_state_store: object | None = None,
    observability_export: ObservabilityExportOperatorConfig | None = None,
) -> FastAPI:
    settings = settings or GovernedContractorBackendSettings.from_env()
    resolved_host_runtime = host_runtime or GovernedContractorHostRuntimeComposition()
    api_key_config = ApiKeyConfig(keys=settings.api_keys_map) if settings.api_keys_map else None

    manifest = build_governed_contractor_manifest()
    env = manifest.environment or build_governed_contractor_environment_profile(settings)
    production_mode = env.execution_mode.value == "strict"
    manifest_for_runtime = manifest
    profile_persistence_kwargs: dict[str, object] = {}
    resolved_tenant_id = manifest.app_id
    strict_topology_reliability = production_mode and process_composition is not None
    provider_invocation_store = (
        process_composition.provider_invocation_store
        if process_composition is not None
        else None
    )
    if strict_topology_reliability and provider_invocation_store is None:
        raise HarnessHostOrchestrationTopologyReliabilityCompositionError(
            "strict governed contractor production host requires "
            "ProductionProcessComposition.provider_invocation_store",
        )
    if process_composition is not None:
        if production_mode:
            env = resolve_reference_production_strict_host_environment(env)
        manifest_for_runtime = manifest.model_copy(update={"environment": env})
        profile_persistence_kwargs = resolve_harness_host_profile_persistence_kwargs_from_composition(
            production_mode=production_mode,
            composition=process_composition,
        )
    elif document_store is not None:
        profile_persistence_kwargs = {"document_store": document_store}
        if key_value_cache is not None:
            profile_persistence_kwargs["key_value_cache"] = key_value_cache
        if production_mode:
            env = resolve_reference_production_strict_host_environment(env)
            manifest_for_runtime = manifest.model_copy(update={"environment": env})
    collaborative_work_integration_profile = None
    if document_store is not None:
        collaborative_work_integration_profile = (
            resolve_governed_contractor_collaborative_work_integration_profile(
                manifest_for_runtime,
                trace_db_path=trace_db_path,
            )
        )
    runtime = build_harness_host_runtime(
        manifest_for_runtime,
        env,
        settings=settings,
        tenant_id=resolved_tenant_id,
        trace_db_path=trace_db_path,
        runtime_events_db_path=runtime_events_db_path,
        checkpoints_db_path=checkpoints_db_path,
        registry_projection=registry_projection,
        collaborative_work_repositories=resolved_host_runtime.collaborative_work_repositories,
        collaborative_work_integration_profile=collaborative_work_integration_profile,
        orchestration_decision_requirement_policy=(
            resolved_host_runtime.decision_requirement_policy
            or default_governed_contractor_harness_orchestration_decision_requirement_policy()
        ),
        runtime_policy_evaluator=resolve_production_runtime_policy_bundle_evaluator(
            settings,
        ),
        active_execution_task_scope=resolved_host_runtime.active_execution_task_scope,
        execution_continuation_state_store=execution_continuation_state_store,
        provider_invocation_store=provider_invocation_store,
        require_strict_orchestration_topology_reliability=strict_topology_reliability,
        strict_orchestration_topology_submission_port_builder=(
            build_governed_contractor_production_orchestration_topology_submission_port
            if strict_topology_reliability
            else None
        ),
        **profile_persistence_kwargs,
    )
    host_execution = runtime.execution
    registry = runtime.registry
    platform = bootstrap_harness_host_platform(runtime)
    if observability_export is not None and observability_export.enabled:
        export_plugin = build_observability_export_runtime_plugin(observability_export)
        if export_plugin is not None:
            export_bootstrap = bootstrap_harness_host_application_plugins(
                runtime,
                [export_plugin],
            )
            platform.shutdown_callbacks.extend(export_bootstrap.shutdown_callbacks)
    checkpoint_store = open_default_task_checkpoint_persistence(db_path=checkpoints_db_path)
    task_enricher = build_reliability_task_enricher(env)
    scheduler_wiring = wire_harness_host_long_running_scheduler(
        runtime,
        checkpoint_store=checkpoint_store,
        host_execution=host_execution,
        task_enricher=task_enricher,
        notification_adapter=None,
        poll_interval_seconds=settings.scheduler_poll_seconds,
        enabled=settings.include_scheduler,
    )
    interaction_service = wire_interaction_intake_service(
        interaction_surface=settings.interaction_surface,
        task_executor=HostTaskExecutionExecutor(host_execution),
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
        host_execution=host_execution,
        registry=runtime.registry,
        prefix=settings.route_prefix,
        default_agent_id=settings.default_agent_id,
    )

    wire_harness_product_observability_dashboard(app, runtime=runtime)

    if settings.include_task_control:
        wire_harness_task_control(
            app,
            enabled=True,
            host_execution=host_execution,
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
    app.state.harness_runtime = runtime
    app.state.governed_contractor_host_runtime = resolved_host_runtime
    return app
