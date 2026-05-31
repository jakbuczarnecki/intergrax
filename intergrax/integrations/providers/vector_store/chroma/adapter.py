# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Chroma vector store adapter — catalog facade delegating to ``rag/``."""

from __future__ import annotations

from intergrax.integrations._shared.vector_store_bridge import VectorStoreBridge
from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations.providers.vector_store.chroma.config import ChromaIntegrationConfig


class ChromaVectorStoreIntegration(VectorStoreBridge):
    """Catalog wrapper over RAG ``ChromaVectorStore``."""

    def __init__(self, config: ChromaIntegrationConfig, inner: VectorStore) -> None:
        super().__init__(config, inner)

    @property
    def config(self) -> ChromaIntegrationConfig:
        return self._config  # type: ignore[return-value]
