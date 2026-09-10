# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Public Qdrant logical point ID normalization."""

from __future__ import annotations

from intergrax.integrations.providers.vector_store.qdrant.rag_store import _normalize_point_id


def normalize_qdrant_logical_point_id(raw_id: str) -> str | int:
    """Map a logical point id to a Qdrant-compatible physical point id."""
    return _normalize_point_id(raw_id)


__all__ = ["normalize_qdrant_logical_point_id"]
