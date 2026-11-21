# © Artur Czarnecki. All rights reserved.

"""Explicit laboratory opt-ins for Tier-0 RAG diagnostic bootstraps (FC-3)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import LabSelectorOptIn
from intergrax.rag.embedding.registry.provider_authority import lab_embedding_provider_slug
from intergrax.rag.vectorstore.bootstrap.integration_vectorstore import lab_inmemory_vector_store_slug

RAG_PLUGIN_EVIDENCE_EMBEDDING_OPT_IN = LabSelectorOptIn(
    selector=lab_embedding_provider_slug(),
    reason="RAG plugin load evidence laboratory embedding bootstrap",
)
RAG_PLUGIN_EVIDENCE_VECTORSTORE_OPT_IN = LabSelectorOptIn(
    selector=lab_inmemory_vector_store_slug(),
    reason="RAG plugin load evidence laboratory vector store bootstrap",
)
