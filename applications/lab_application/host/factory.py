# © Artur Czarnecki. All rights reserved.

"""Assemble lab execution routes + debug inspection API (Phase L.3)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI

from intergrax.debug.app import create_debug_app
from intergrax.debug.hitl_service import DebugHitlResumeService
from intergrax.debug.interaction_service import DebugInteractionIntakeService
from intergrax.debug.store import open_default_task_checkpoint_persistence
from intergrax.runtime.interactions.verification.factory import create_inbound_verifier
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore
from intergrax.runtime.registry.agent_registry import AgentRegistry
from lab_application.host.settings import LabApplicationSettings
from lab_application.host.wiring import build_lab_registry
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

    Combines:

    - ``POST /v1/lab/run`` — execute arbitrary registered agents via UnifiedTaskRunner
    - ``GET /v1/lab/agents`` — list active agents and capabilities
    - ``/debug/*`` — trace, events, checkpoints, progress, experiments, HITL intake
    """
    settings = settings or LabApplicationSettings.from_env()
    resolved_registry = registry or build_lab_registry(settings=settings)
    checkpoint_store = open_default_task_checkpoint_persistence(db_path=checkpoints_db_path)
    trace_store = InMemoryRunTraceStore()
    nexus_loop = NexusLoop(
        resolved_registry,
        checkpoint_store=checkpoint_store,
        trace_store=trace_store,
    )
    interaction_service = DebugInteractionIntakeService(
        nexus_loop=nexus_loop,
        verifier=create_inbound_verifier(),
    )
    hitl_service = DebugHitlResumeService(
        resolved_registry,
        checkpoint_store=checkpoint_store,
    )

    app = create_debug_app(
        db_path=db_path,
        experiments_db_path=experiments_db_path,
        runtime_events_db_path=runtime_events_db_path,
        checkpoints_db_path=checkpoints_db_path,
        registry=resolved_registry,
        nexus_loop=nexus_loop,
        interaction_service=interaction_service,
        hitl_service=hitl_service,
        checkpoint_store=checkpoint_store,
    )
    app.title = "Intergrax Lab Application"
    app.description = (
        "Agent OS experimentation environment — run agents, inspect traces, "
        "checkpoints, runtime events, and experiments (Phase L.3)."
    )
    mount_lab_routes(app, nexus_loop=nexus_loop, prefix=settings.route_prefix)
    return app
