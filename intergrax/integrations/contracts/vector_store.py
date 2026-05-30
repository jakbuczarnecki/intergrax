# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Vector store integration contract — re-export from ``intergrax.rag`` (Phase M.6 P2)."""

from intergrax.rag.vectorstore.contracts.vector_store import (
    MetadataFilter,
    VectorStore,
    VectorStoreHit,
)

__all__ = [
    "MetadataFilter",
    "VectorStore",
    "VectorStoreHit",
]
