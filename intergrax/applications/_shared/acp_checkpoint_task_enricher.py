# © Artur Czarnecki. All rights reserved.

"""Task enricher — inject ACP agent checkpoint store for Tier-3 hosts."""

from __future__ import annotations

from collections.abc import Callable

from intergrax.agents.persistence.checkpoint_wiring import (
    attach_checkpoint_wiring,
    should_resume_acp_checkpoint,
)
from intergrax.agents.persistence.checkpoint_store import AgentCheckpointStore
from intergrax.contracts.acp_metadata_keys import AcpMetadataKey
from intergrax.runtime.task.task import Task


def make_acp_checkpoint_task_enricher(
    store: AgentCheckpointStore | None,
) -> Callable[[Task], Task] | None:
    """Build a task enricher that wires ``AgentCheckpointStore`` into task metadata."""
    if store is None:
        return None

    def enricher(task: Task) -> Task:
        run_id = task.task_id
        metadata = dict(task.metadata)
        metadata.setdefault("user_id", task.user_id)
        metadata.setdefault("run_id", run_id)
        metadata.setdefault("task_id", run_id)
        resume = should_resume_acp_checkpoint(
            metadata,
            store=store,
            run_id=run_id,
            tenant_id=task.tenant_id,
        )
        wired = attach_checkpoint_wiring(metadata, store, resume=resume)
        wired[AcpMetadataKey.SESSION_ENABLED] = True
        return task.model_copy(update={"metadata": wired})

    return enricher
