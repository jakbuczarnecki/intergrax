# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Qdrant vector store adapter — catalog facade delegating to ``rag/``."""

from __future__ import annotations

from intergrax.integrations._shared.vector_store_bridge import VectorStoreBridge
from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations.providers.vector_store.qdrant.config import QdrantIntegrationConfig


class QdrantVectorStoreIntegration(VectorStoreBridge):
    """Catalog wrapper over RAG ``QdrantVectorStore``."""

    def __init__(self, config: QdrantIntegrationConfig, inner: VectorStore) -> None:
        super().__init__(config, inner)

    @property
    def config(self) -> QdrantIntegrationConfig:
        return self._config  # type: ignore[return-value]
