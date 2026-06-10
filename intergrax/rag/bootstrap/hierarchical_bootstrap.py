# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""TOC / dual-index bootstrap helpers (M-RAG.24)."""

from __future__ import annotations

from typing import Optional

from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.vectorstore.bootstrap.integration_vectorstore import create_vectorstore_manager
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager


def profile_uses_hierarchical_index(profile: RagProfile) -> bool:
    """True when dual-index ingest and TOC retriever wiring should activate."""
    if profile.hierarchical_index_enabled:
        return True
    retriever_ids = {
        profile.retriever_id,
        profile.fast_retriever_id,
        profile.deep_retriever_id,
    }
    return "hierarchical" in retriever_ids


def create_toc_vectorstore_manager(
    *,
    integration_profile: Optional[IntegrationProfile] = None,
    tenant_id: Optional[str] = None,
    chunks_store: Optional[BaseVectorstoreManager] = None,
) -> BaseVectorstoreManager:
    """Sibling vector store for TOC entries (separate collection when backend supports it)."""
    overrides: dict[str, object] = {}
    if chunks_store is not None:
        try:
            names = list(chunks_store.list_collections())
        except Exception:
            names = []
        if names:
            overrides["collection_name"] = f"{names[0]}-toc"
    return create_vectorstore_manager(
        profile=integration_profile,
        tenant_id=tenant_id,
        **overrides,
    )


def resolve_toc_vectorstore_for_profile(
    profile: RagProfile,
    *,
    integration_profile: Optional[IntegrationProfile] = None,
    tenant_id: Optional[str] = None,
    chunks_store: Optional[BaseVectorstoreManager] = None,
) -> Optional[BaseVectorstoreManager]:
    if not profile_uses_hierarchical_index(profile):
        return None
    return create_toc_vectorstore_manager(
        integration_profile=integration_profile,
        tenant_id=tenant_id,
        chunks_store=chunks_store,
    )
