# © Artur Czarnecki. All rights reserved.

"""Tier-3 ACP agent checkpoint store resolution (ACP-CLOSE-PROD-1)."""

from __future__ import annotations

from pathlib import Path

from intergrax.agents.persistence.checkpoint_wiring import open_agent_checkpoint_store
from intergrax.agents.persistence.checkpoint_store import AgentCheckpointStore
from intergrax.agents.persistence.compensation_queue_store import CompensationQueueStore
from intergrax.agents.persistence.compensation_queue_wiring import open_compensation_queue_store
from intergrax.debug.store import resolve_task_checkpoints_db_path


def _checkpoints_parent(checkpoints_db_path: Path | None = None) -> Path | None:
    base = checkpoints_db_path or resolve_task_checkpoints_db_path(None)
    if base is None:
        return None
    return base.parent


def resolve_agent_checkpoint_db_path(
    checkpoints_db_path: Path | None = None,
) -> Path | None:
    """Derive durable agent checkpoint DB path adjacent to Nexus task checkpoints."""
    parent = _checkpoints_parent(checkpoints_db_path)
    if parent is None:
        return None
    return parent / "agent_checkpoints.db"


def resolve_compensation_queue_db_path(
    checkpoints_db_path: Path | None = None,
) -> Path | None:
    """Derive durable compensation queue DB path adjacent to Nexus task checkpoints."""
    parent = _checkpoints_parent(checkpoints_db_path)
    if parent is None:
        return None
    return parent / "compensation_queue.db"


def resolve_host_agent_checkpoint_store(
    *,
    agent_checkpoint_store: AgentCheckpointStore | None = None,
    checkpoints_db_path: Path | None = None,
) -> AgentCheckpointStore:
    """Materialize host ``AgentCheckpointStore`` (SQLite when path known, else in-memory)."""
    if agent_checkpoint_store is not None:
        return agent_checkpoint_store
    return open_agent_checkpoint_store(resolve_agent_checkpoint_db_path(checkpoints_db_path))


def resolve_host_compensation_queue_store(
    *,
    compensation_queue_store: CompensationQueueStore | None = None,
    checkpoints_db_path: Path | None = None,
) -> CompensationQueueStore:
    """Materialize host ``CompensationQueueStore`` (SQLite when path known, else in-memory)."""
    if compensation_queue_store is not None:
        return compensation_queue_store
    return open_compensation_queue_store(resolve_compensation_queue_db_path(checkpoints_db_path))
