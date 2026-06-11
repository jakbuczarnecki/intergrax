# © Artur Czarnecki. All rights reserved.

"""Host wiring for durable compensation queue (ACP-CLOSE-PROD-5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from intergrax.agents.persistence.compensation_queue_store import (
    CompensationQueueStore,
    InMemoryCompensationQueueStore,
    SQLiteCompensationQueueStore,
)
from intergrax.contracts.acp_metadata_keys import AcpMetadataKey


def open_compensation_queue_store(
    db_path: str | Path | None = None,
) -> CompensationQueueStore:
    """Open durable compensation queue; in-memory when no path."""
    if db_path is None:
        return InMemoryCompensationQueueStore()
    return SQLiteCompensationQueueStore(db_path)


def attach_compensation_queue_wiring(
    metadata: dict[str, Any],
    store: CompensationQueueStore,
) -> dict[str, Any]:
    wired = dict(metadata)
    wired[AcpMetadataKey.COMPENSATION_QUEUE_STORE] = store
    return wired


def resolve_compensation_queue_from_metadata(
    metadata: dict[str, Any],
) -> CompensationQueueStore | None:
    store = metadata.get(AcpMetadataKey.COMPENSATION_QUEUE_STORE)
    if isinstance(store, CompensationQueueStore):
        return store
    return None


def inject_acp_compensation_queue_metadata(
    metadata: dict[str, Any],
    store: CompensationQueueStore | None,
) -> None:
    if store is None:
        return
    if not metadata.get(AcpMetadataKey.SESSION_ENABLED):
        return
    metadata[AcpMetadataKey.COMPENSATION_QUEUE_STORE] = store


def make_acp_compensation_queue_task_enricher(
    store: CompensationQueueStore | None,
):
    """Build a task enricher that wires compensation queue into task metadata."""
    from collections.abc import Callable

    from intergrax.runtime.task.task import Task

    if store is None:
        return None

    def enricher(task: Task) -> Task:
        metadata = dict(task.metadata)
        inject_acp_compensation_queue_metadata(metadata, store)
        return task.model_copy(update={"metadata": metadata})

    return enricher
