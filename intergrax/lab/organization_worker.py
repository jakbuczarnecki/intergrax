# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Laboratory wiring for §38 Organization Worker demo (Phase H.6)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from intergrax.debug.app import create_debug_app
from intergrax.debug.hitl_service import DebugHitlResumeService
from intergrax.debug.interaction_service import DebugInteractionIntakeService
from intergrax.debug.store import open_default_task_checkpoint_persistence
from intergrax.runtime.interactions.metadata_keys import INTERACTION_CHANNEL_KEY
from intergrax.runtime.interactions.verification.factory import create_inbound_verifier
from intergrax.runtime.long_running.notification import NotificationAdapter
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.registry.bootstrap import build_organization_worker_registry
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_contract import TaskLongRunningOptions

ORG_WORKER_CAPABILITY = "org.vendor_report"

_NOTIFY_CHANNEL_BY_INTERACTION = {
    "slash_command": "slack",
    "teams": "teams",
    "lab": "log",
}


def enrich_organization_worker_task(task: Task) -> Task:
    """
    Enable long-running checkpoints + channel notifications for org.* capabilities.

    Maps interaction surface (Slack slash, Teams activity, lab JSON) to notify_channel.
    """
    capability = task.context.capability or ""
    if not capability.startswith("org."):
        return task

    interaction_channel = str(task.metadata.get(INTERACTION_CHANNEL_KEY) or "lab")
    notify_channel = _NOTIFY_CHANNEL_BY_INTERACTION.get(interaction_channel, "log")
    explicit = task.metadata.get("notify_channel")
    if isinstance(explicit, str) and explicit.strip():
        notify_channel = explicit.strip()

    task.options = task.options.model_copy(
        update={
            "long_running": TaskLongRunningOptions(
                enabled=True,
                checkpoint_on_pause=True,
                notify_channel=notify_channel,
            )
        },
        deep=True,
    )
    task.sync_metadata()
    return task


def create_organization_worker_lab_app(
    *,
    db_path: Path | None = None,
    experiments_db_path: Path | None = None,
    runtime_events_db_path: Path | None = None,
    checkpoints_db_path: Path | None = None,
    registry: Optional[AgentRegistry] = None,
    notification_adapter: Optional[NotificationAdapter] = None,
    task_enricher: Optional[Callable[[Task], Task]] = enrich_organization_worker_task,
):
    """
    Debug API pre-wired for Organization Worker (§38).

    Intake → Nexus → HITL pause → notification → ``POST …/human-response`` resume.
    """
    resolved_registry = registry or build_organization_worker_registry()
    checkpoint_store = open_default_task_checkpoint_persistence(db_path=checkpoints_db_path)
    nexus_loop = NexusLoop(
        resolved_registry,
        checkpoint_store=checkpoint_store,
        notification_adapter=notification_adapter,
    )
    interaction_service = DebugInteractionIntakeService(
        nexus_loop=nexus_loop,
        verifier=create_inbound_verifier(),
        task_enricher=task_enricher,
    )
    hitl_service = DebugHitlResumeService(
        resolved_registry,
        checkpoint_store=checkpoint_store,
    )
    return create_debug_app(
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
