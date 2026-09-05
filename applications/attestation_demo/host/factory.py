# © Artur Czarnecki. All rights reserved.

"""Assemble lab routes + debug API for attestation_demo (partner PoC)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI

from intergrax.applications._shared.attestation_runtime_bridge import build_boundary_event_buffer
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.workspace_cleanup_wiring import (
    apply_factory_lifespans,
    build_factory_lifespans,
)
from intergrax.applications._shared.platform_wiring import bootstrap_nexus_platform
from intergrax.applications._shared.plugin_bootstrap import attach_plugin_shutdown
from intergrax.applications._shared.task_control_wiring import (
    build_reliability_task_enricher,
    build_task_runner_with_enricher,
    wire_harness_task_control,
)
from intergrax.debug.app import create_debug_app
from intergrax.debug.hitl_service import DebugHitlResumeService
from intergrax.debug.store import open_default_task_checkpoint_persistence
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.attestation.buffer import BoundaryEventBuffer
from intergrax.applications._shared.host_task_execution_wiring import build_environment_host_task_execution
from intergrax.applications._shared.harness_host_runtime_compat import resolve_harness_host_nexus_loop_legacy
from attestation_demo.host.agent_builders import ATTESTATION_DEMO_AGENT_BUILDERS
from attestation_demo.host.integration_wiring import wire_attestation_demo_integrations
from attestation_demo.host.settings import AttestationDemoSettings
from attestation_demo.manifest import build_attestation_demo_manifest
from attestation_demo.serving.fastapi_router import mount_attestation_demo_routes


def create_attestation_demo_application(
    *,
    settings: Optional[AttestationDemoSettings] = None,
    db_path: Path | None = None,
    runtime_events_db_path: Path | None = None,
    checkpoints_db_path: Path | None = None,
    document_store: InMemoryDocumentStore | None = None,
    boundary_event_buffer: BoundaryEventBuffer | None = None,
) -> FastAPI:
    settings = settings or AttestationDemoSettings.from_env()
    integrations = wire_attestation_demo_integrations(
        db_path=db_path,
        document_store=document_store,
    )
    resolved_document_store = integrations.document_store
    manifest = build_attestation_demo_manifest()
    env = manifest.resolved_environment()
    resolved_buffer = boundary_event_buffer or build_boundary_event_buffer(env) or BoundaryEventBuffer()
    runtime = build_harness_host_runtime(
        manifest.model_copy(update={"environment": env}),
        env,
        settings=settings,
        trace_db_path=db_path,
        runtime_events_db_path=runtime_events_db_path,
        use_in_memory_trace=db_path is None,
        checkpoints_db_path=checkpoints_db_path,
        builders=ATTESTATION_DEMO_AGENT_BUILDERS,
        document_store=resolved_document_store,
        boundary_event_buffer=resolved_buffer,
    )
    host_execution = runtime.execution
    nexus_loop = resolve_harness_host_nexus_loop_legacy(runtime)
    resolved_registry = runtime.registry
    platform = bootstrap_nexus_platform(
        nexus_loop,
        trace_store=runtime.observability.trace_store,  # type: ignore[arg-type]
    )
    checkpoint_store = open_default_task_checkpoint_persistence(db_path=checkpoints_db_path)
    task_enricher = build_reliability_task_enricher(
        env,
        agent_checkpoint_store=runtime.agent_checkpoint_store,
        compensation_queue_store=runtime.compensation_queue_store,
        idempotency_store=runtime.reliability.idempotency_store,
    )
    task_runner = build_task_runner_with_enricher(nexus_loop, task_enricher)
    hitl_service = DebugHitlResumeService(
        resolved_registry,
        checkpoint_store=checkpoint_store,
    )
    app = create_debug_app(
        db_path=runtime.observability.trace_db_path,
        runtime_events_db_path=runtime.observability.runtime_events_db_path,
        checkpoints_db_path=checkpoints_db_path,
        registry=resolved_registry,
        nexus_loop=nexus_loop,
        hitl_service=hitl_service,
        checkpoint_store=checkpoint_store,
        trace_store=runtime.observability.trace_store,
        runtime_event_store=runtime.observability.runtime_event_store,
    )
    app.title = "Intergrax Attestation Demo (Partner PoC)"
    mount_attestation_demo_routes(
        app,
        host_execution=host_execution,
        registry=resolved_registry,
        boundary_event_buffer=runtime.boundary_event_buffer or resolved_buffer,
        prefix=settings.route_prefix,
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
    apply_factory_lifespans(app, runtime, schedulers=None)
    attach_plugin_shutdown(app, platform.shutdown_callbacks)
    return app
