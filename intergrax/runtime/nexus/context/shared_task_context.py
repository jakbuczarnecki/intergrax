# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cross-agent shared task payload (architecture §42.14.1, Phase I.3)."""

from __future__ import annotations

from typing import Any, Dict, Optional  # noqa: F401, UP035

from pydantic import BaseModel, Field

from intergrax.runtime.nexus.artifacts.models import TraceArtifactRef
from intergrax.runtime.task.task_metadata_keys import TaskMetadataKey

DEFAULT_SHARED_MEMORY_NAMESPACE = "shared"


class SharedArtifactEntry(BaseModel):
    """Lightweight artifact reference keyed in ``SharedTaskContext.artifacts``."""

    artifact_id: str
    kind: str
    size_bytes: int = 0

    @classmethod
    def from_ref(cls, ref: TraceArtifactRef) -> SharedArtifactEntry:
        return cls(
            artifact_id=ref.artifact_id,
            kind=ref.kind,
            size_bytes=ref.size_bytes,
        )


class SharedTaskContext(BaseModel):
    """
    Nexus-owned cross-agent payload for one task.

    Writes MUST go through ``ContextManager`` — not agent globals or direct stores.
    """

    task_id: str
    artifacts: dict[str, SharedArtifactEntry] = Field(default_factory=dict)
    structured_outputs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    entry_versions: dict[str, int] = Field(default_factory=dict)
    updated_by: dict[str, str] = Field(default_factory=dict)
    memory_namespace: str = DEFAULT_SHARED_MEMORY_NAMESPACE
    version: int = 1
    schema_version: str = "shared_task_context.v1"


class SharedContextConflictError(ValueError):
    """Optimistic concurrency failure on ``SharedTaskContext.version``."""


def load_shared_task_context(task_or_metadata: Any) -> SharedTaskContext | None:
    """Load shared context from ``Task`` or a metadata mapping."""
    if isinstance(task_or_metadata, SharedTaskContext):
        return task_or_metadata
    metadata = task_or_metadata.metadata if hasattr(task_or_metadata, "metadata") else task_or_metadata
    if not isinstance(metadata, dict):
        return None
    return load_shared_task_context_from_metadata(metadata)


def load_shared_task_context_from_metadata(
    metadata: dict[str, Any],
) -> SharedTaskContext | None:
    raw = metadata.get(TaskMetadataKey.SHARED_TASK_CONTEXT)
    if raw is None:
        return None
    if isinstance(raw, SharedTaskContext):
        return raw
    if isinstance(raw, dict):
        return SharedTaskContext.model_validate(raw)
    return None


def save_shared_task_context(task_or_metadata: Any, shared: SharedTaskContext) -> None:
    """Persist shared context on ``Task.metadata`` or a metadata dict."""
    payload = shared.model_dump(mode="json")
    if hasattr(task_or_metadata, "metadata"):
        task_or_metadata.metadata[TaskMetadataKey.SHARED_TASK_CONTEXT] = payload
        return
    if isinstance(task_or_metadata, dict):
        task_or_metadata[TaskMetadataKey.SHARED_TASK_CONTEXT] = payload


def get_or_create_shared_task_context(task_or_metadata: Any, *, task_id: str) -> SharedTaskContext:
    existing = load_shared_task_context(task_or_metadata)
    if existing is not None:
        return existing
    return SharedTaskContext(task_id=task_id)


__all__ = [
    "DEFAULT_SHARED_MEMORY_NAMESPACE",
    "TaskMetadataKey",
    "SharedArtifactEntry",
    "SharedContextConflictError",
    "SharedTaskContext",
    "get_or_create_shared_task_context",
    "load_shared_task_context",
    "load_shared_task_context_from_metadata",
    "save_shared_task_context",
]
