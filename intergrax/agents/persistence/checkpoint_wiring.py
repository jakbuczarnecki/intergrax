# © Artur Czarnecki. All rights reserved.

"""Host wiring for ACP checkpoint store injection (ACP-PROD-1 depth)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from intergrax.agents.persistence.checkpoint_store import (
    AgentCheckpointStore,
    InMemoryAgentCheckpointStore,
    SQLiteAgentCheckpointStore,
)
from intergrax.contracts.acp_metadata_keys import AcpMetadataKey
from intergrax.contracts.agent_run import AgentRunRequest


def open_agent_checkpoint_store(
  db_path: str | Path | None = None,
) -> AgentCheckpointStore:
    """Open durable agent checkpoint store; in-memory when no path."""
    if db_path is None:
        return InMemoryAgentCheckpointStore()
    return SQLiteAgentCheckpointStore(db_path)


def attach_checkpoint_wiring(
    metadata: dict[str, Any],
    store: AgentCheckpointStore,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Return metadata with checkpoint store (and optional resume flag) attached."""
    wired = dict(metadata)
    wired[AcpMetadataKey.CHECKPOINT_STORE] = store
    if resume:
        wired[AcpMetadataKey.RESUME_FROM_CHECKPOINT] = True
    return wired


def wire_acp_run_request(
    request: AgentRunRequest,
    store: AgentCheckpointStore,
    *,
    resume: bool = False,
) -> AgentRunRequest:
    """Attach checkpoint store to a typed ``AgentRunRequest``."""
    return request.model_copy(
        update={
            "metadata": attach_checkpoint_wiring(
                dict(request.metadata),
                store,
                resume=resume,
            ),
        },
    )


def should_resume_acp_checkpoint(
    metadata: dict[str, Any],
    *,
    store: AgentCheckpointStore | None,
    run_id: str,
    tenant_id: str,
) -> bool:
    """Whether an ACP session should load the latest agent checkpoint."""
    if metadata.get(AcpMetadataKey.RESUME_FROM_CHECKPOINT) in {True, "true", "1", 1}:
        return True
    if metadata.get("human_response") is not None:
        return True
    if store is None:
        return False
    return store.get_latest(run_id, tenant_id) is not None


def inject_acp_checkpoint_metadata(
    metadata: dict[str, Any],
    *,
    store: AgentCheckpointStore | None,
    run_id: str,
    tenant_id: str,
) -> None:
    """Mutate runtime/task metadata in place when ACP session is enabled."""
    if store is None:
        return
    if not metadata.get(AcpMetadataKey.SESSION_ENABLED):
        return
    metadata.setdefault("run_id", run_id)
    metadata.setdefault("task_id", run_id)
    metadata[AcpMetadataKey.CHECKPOINT_STORE] = store
    if should_resume_acp_checkpoint(
        metadata,
        store=store,
        run_id=run_id,
        tenant_id=tenant_id,
    ):
        metadata[AcpMetadataKey.RESUME_FROM_CHECKPOINT] = True
