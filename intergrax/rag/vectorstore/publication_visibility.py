# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared publication-generation visibility law for vector-store retrieval."""

from __future__ import annotations

from intergrax.distributed.source_operation import (
    RagSourceOperationKey,
    SourceOperationCoordinator,
)


def vector_record_visible(
    *,
    record_generation: str | None,
    source_key: RagSourceOperationKey,
    coordinator: SourceOperationCoordinator | None,
) -> bool:
    """Return whether one vector record is visible under the canonical law."""
    if record_generation is None:
        return True
    if coordinator is None:
        return False
    try:
        active_generation = coordinator.active_publication_generation(key=source_key)
    except Exception:
        return False
    if active_generation is None:
        return False
    return active_generation == record_generation
