# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Standalone FastAPI app for trace inspection and experiment registry (Phase D.2–D.3)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI

from intergrax.debug.hitl_service import DebugHitlResumeService
from intergrax.debug.interaction_service import DebugInteractionIntakeService
from intergrax.debug.router import create_debug_router
from intergrax.debug.store import (
    open_default_task_checkpoint_persistence,
    open_runtime_event_persistence,
)
from intergrax.fastapi_core.routers.health import health_router
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.interactions.verification.factory import create_inbound_verifier
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry


def create_debug_app(
    *,
    db_path: Path | None = None,
    experiments_db_path: Path | None = None,
    runtime_events_db_path: Path | None = None,
    checkpoints_db_path: Path | None = None,
    runtime_event_store: RuntimeEventPersistence | None = None,
    checkpoint_store: TaskCheckpointPersistence | None = None,
    registry: Optional[AgentRegistry] = None,
    hitl_service: DebugHitlResumeService | None = None,
    interaction_service: DebugInteractionIntakeService | None = None,
    nexus_loop: NexusLoop | None = None,
) -> FastAPI:
    """
    Laboratory debug API over trace, runtime events, checkpoints, and experiments.

    Persistence backends are injectable (``runtime_event_store``, ``checkpoint_store``)
    or resolved from explicit paths / ``INTERGRAX_*`` env vars.
    """
    resolved_checkpoint_store = open_default_task_checkpoint_persistence(
        db_path=checkpoints_db_path,
        implementation=checkpoint_store,
    )
    resolved_runtime_store = open_runtime_event_persistence(
        db_path=runtime_events_db_path,
        implementation=runtime_event_store,
    )
    resolved_hitl = hitl_service
    if resolved_hitl is None and registry is not None:
        resolved_hitl = DebugHitlResumeService(
            registry,
            checkpoint_store=resolved_checkpoint_store,
            runtime_event_store=resolved_runtime_store,
        )

    resolved_loop = nexus_loop
    if resolved_loop is None and registry is not None:
        resolved_loop = NexusLoop(
            registry,
            checkpoint_store=resolved_checkpoint_store,
            runtime_event_store=resolved_runtime_store,
        )

    resolved_interaction = interaction_service
    if resolved_interaction is None and registry is not None:
        resolved_interaction = DebugInteractionIntakeService(
            nexus_loop=resolved_loop,
            verifier=create_inbound_verifier(),
        )

    app = FastAPI(
        title="Intergrax Debug API",
        version="0.2.0",
        description=(
            "Inspect Nexus runs/traces, runtime events, checkpoints, "
            "and manage experiments (Phase D + G.6, §19, §35)."
        ),
    )
    app.include_router(health_router)
    app.include_router(
        create_debug_router(
            db_path=db_path,
            experiments_db_path=experiments_db_path,
            runtime_events_db_path=runtime_events_db_path,
            checkpoints_db_path=checkpoints_db_path,
            runtime_event_store=runtime_event_store,
            checkpoint_store=resolved_checkpoint_store,
            hitl_service=resolved_hitl,
            interaction_service=resolved_interaction,
        )
    )
    return app
