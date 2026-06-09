# © Artur Czarnecki. All rights reserved.

"""Assemble lab execution routes + debug inspection API (Phase L.3)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI

from intergrax.debug.app import create_debug_app
from intergrax.llm_adapters.tracking.exposition import register_llm_metrics_routes
from intergrax.debug.hitl_service import DebugHitlResumeService
from intergrax.debug.interaction_service import DebugInteractionIntakeService
from intergrax.runtime.interactions.router import create_interaction_intake_router
from intergrax.runtime.interactions.verification.factory import create_inbound_verifier
from intergrax.runtime.long_running.wiring import wire_long_running_scheduler
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
from intergrax.applications._shared.fastapi_mcp import (
    apply_lifespans,
    couple_fastapi_with_mcp,
    make_scheduler_lifespan,
)
from lab_application.host.settings import LabApplicationSettings
from lab_application.host.tool_wiring import wire_lab_tools
from lab_application.host.wiring import bootstrap_lab_integration_wiring, build_lab_registry
from lab_application.mcp.server import build_lab_mcp_server
from intergrax.applications._shared.task_defaults import make_lab_harness_task_enricher
from intergrax.applications._shared.platform_wiring import bootstrap_nexus_platform
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.lab_environment_profile import build_lab_environment_profile
from lab_application.manifest import build_lab_manifest
from intergrax.applications._shared.plugin_bootstrap import attach_plugin_shutdown
from intergrax.applications._shared.identity_wiring import wire_application_identity
from intergrax.applications._shared.harness_auth import (
    require_harness_auth,
    resolve_harness_api_key,
)
from intergrax.runtime.adaptive.proposal_store import SQLiteProposalStore, default_proposal_store_path
from intergrax.runtime.adaptive.signal_store import SQLiteSignalStore, default_signal_store_path
from lab_application.serving.fastapi_router import mount_lab_routes
from intergrax.applications._shared.harness_task_routes import mount_harness_task_routes
from intergrax.applications._shared.reliability_wiring import apply_reliability_task_defaults


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

    Integrations use ``IntegrationProfile.lab_harness_preset()`` (Phase T-Ops.1):
    sqlite, log, lab_json, OTEL (disable via ``LAB_OTEL_ENABLED=false``).

    Combines:

    - ``POST /v1/lab/run`` — execute arbitrary registered agents via UnifiedTaskRunner
    - ``GET /v1/lab/agents`` — list active agents and capabilities
    - ``POST /v1/interactions/intake`` — production Slack / Teams / lab inbound webhooks (B.12)
    - Long-running scheduler — HITL timeout / delayed resume (B.05)
    - ``/debug/*`` — trace, events, checkpoints, progress, experiments, HITL intake
    """
    settings = settings or LabApplicationSettings.from_env()
    if settings.requires_harness_api_key and resolve_harness_api_key() is None:
        raise ValueError(
            "INTERGRAX_HARNESS_API_KEY is required when INTERGRAX_ENV is stage/prod "
            "or LAB_STRICT_HARNESS=true (Phase W-OPS.7)."
        )
    integrations = bootstrap_lab_integration_wiring(
        settings=settings,
        db_path=db_path,
        experiments_db_path=experiments_db_path,
        runtime_events_db_path=runtime_events_db_path,
        checkpoints_db_path=checkpoints_db_path,
        harness=settings.harness,
        otel_enabled=settings.otel_enabled,
    )
    manifest = build_lab_manifest(settings)
    lab_env = manifest.environment or build_lab_environment_profile(settings)
    if manifest.environment is None:
        manifest = manifest.model_copy(update={"environment": lab_env})
    resolved_registry = registry or build_lab_registry(
        settings=settings,
        integration_profile=integrations.profile,
        trace_db_path=integrations.trace_db_path,
    )
    runtime = build_harness_host_runtime(
        manifest,
        lab_env,
        settings=settings,
        trace_db_path=integrations.trace_db_path,
        runtime_events_db_path=integrations.runtime_events_db_path,
        checkpoints_db_path=integrations.checkpoints_db_path,
        registry=resolved_registry,
        checkpoint_store=integrations.checkpoint_store,
        notification_adapter=integrations.notification_adapter,
    )
    nexus_loop = runtime.nexus_loop
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
    lab_notify_enricher = make_lab_harness_task_enricher(
        default_notify_channel=integrations.default_long_running_notify_channel,
        harness=settings.harness,
    )

    def task_enricher(task):
        task = apply_reliability_task_defaults(task, lab_env)
        if lab_notify_enricher is not None:
            task = lab_notify_enricher(task)
        return task
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
        adaptive_signal_store=SQLiteSignalStore(db_path=default_signal_store_path()),
        adaptive_proposal_store=SQLiteProposalStore(db_path=default_proposal_store_path()),
        include_adaptive_debug_routes=lab_env.adaptive_profile.debug_readonly_routes,
        include_integration_health_routes=settings.harness,
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
    mount_harness_task_routes(
        app,
        task_runner=task_runner,
        checkpoint_store=integrations.checkpoint_store,
        task_enricher=task_enricher,
    )
    if settings.include_interaction_routes:
        app.include_router(
            create_interaction_intake_router(
                interaction_service,
                execute_default=True,
            ),
            prefix=settings.interaction_route_prefix,
            dependencies=[Depends(require_harness_auth)],
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
        apply_lifespans(app, make_scheduler_lifespan(scheduler))
    attach_plugin_shutdown(app, plugin_bootstrap.shutdown_callbacks)
    register_llm_metrics_routes(app)
    wire_application_identity(
        app,
        lab_env.identity_profile,
        integration_profile=integrations.profile,
    )
    return app
