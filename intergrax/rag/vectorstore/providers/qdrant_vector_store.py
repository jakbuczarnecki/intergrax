# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Backward-compatible shim — ``qdrant_client`` imports live in Integration Library ``rag_store``."""

from intergrax.integrations.providers.vector_store.qdrant.rag_store import QdrantConfig, QdrantVectorStore

__all__ = ["QdrantConfig", "QdrantVectorStore"]
