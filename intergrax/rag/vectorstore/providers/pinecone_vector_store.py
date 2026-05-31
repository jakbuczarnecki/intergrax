# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Backward-compatible shim — Pinecone SDK imports live in Integration Library ``rag_store``."""

from intergrax.integrations.providers.vector_store.pinecone.rag_store import (
    PineconeConfig,
    PineconeVectorStore,
)

__all__ = ["PineconeConfig", "PineconeVectorStore"]
