# © Artur Czarnecki. All rights reserved.

"""Task enricher — wire ``AgentCheckpointStore`` only for explicit ``acp.session.v1`` tasks."""

from __future__ import annotations

from collections.abc import Callable

from intergrax.agents.persistence.checkpoint_wiring import (
    attach_checkpoint_wiring,
    should_resume_acp_checkpoint,
)
from intergrax.agents.persistence.checkpoint_store import AgentCheckpointStore
from intergrax.contracts.acp_metadata_keys import AcpMetadataKey
from intergrax.runtime.task.task import Task


def _acp_session_enabled_in_metadata(metadata: dict[str, object]) -> bool:
    flag = metadata.get(AcpMetadataKey.SESSION_ENABLED)
    return flag is True or flag == "true" or flag == "1"


def make_acp_checkpoint_task_enricher(
    store: AgentCheckpointStore | None,
) -> Callable[[Task], Task] | None:
    """Build enricher that attaches ``AgentCheckpointStore`` when ``acp.session.v1`` is enabled."""
    if store is None:
        return None

    def enricher(task: Task) -> Task:
        metadata = dict(task.metadata)
        if not _acp_session_enabled_in_metadata(metadata):
            return task
        run_id = task.task_id
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
        return task.model_copy(update={"metadata": wired})

    return enricher
