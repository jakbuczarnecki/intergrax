# © Artur Czarnecki. All rights reserved.

"""Bridge Nexus SharedTaskContext ↔ contracts SharedContextView."""

from __future__ import annotations

from typing import Any

from intergrax.contracts.shared_context import SharedArtifactRef, SharedContextView
from intergrax.runtime.nexus.context.shared_task_context import (
    SharedArtifactEntry,
    SharedTaskContext,
    get_or_create_shared_task_context,
    load_shared_task_context,
    save_shared_task_context,
)


def _artifact_to_ref(entry: SharedArtifactEntry) -> SharedArtifactRef:
    return SharedArtifactRef(
        artifact_id=entry.artifact_id,
        kind=entry.kind,
        size_bytes=entry.size_bytes,
    )


def _ref_to_entry(ref: SharedArtifactRef) -> SharedArtifactEntry:
    return SharedArtifactEntry(
        artifact_id=ref.artifact_id,
        kind=ref.kind,
        size_bytes=ref.size_bytes,
    )


def view_from_backing(backing: SharedTaskContext) -> SharedContextView:
    return SharedContextView(
        task_id=backing.task_id,
        version=backing.version,
        memory_namespace=backing.memory_namespace,
        artifacts={key: _artifact_to_ref(entry) for key, entry in backing.artifacts.items()},
        structured_outputs=dict(backing.structured_outputs),
    )


def view_from_task_metadata(task_or_metadata: Any, *, task_id: str) -> SharedContextView:
    backing = get_or_create_shared_task_context(task_or_metadata, task_id=task_id)
    return view_from_backing(backing)


def load_view(task_or_metadata: Any) -> SharedContextView | None:
    backing = load_shared_task_context(task_or_metadata)
    if backing is None:
        return None
    return view_from_backing(backing)


def persist_view(task_or_metadata: Any, view: SharedContextView) -> None:
    backing = SharedTaskContext(
        task_id=view.task_id,
        version=view.version,
        memory_namespace=view.memory_namespace,
        artifacts={key: _ref_to_entry(ref) for key, ref in view.artifacts.items()},
        structured_outputs=dict(view.structured_outputs),
    )
    save_shared_task_context(task_or_metadata, backing)
