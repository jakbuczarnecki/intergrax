# © Artur Czarnecki. All rights reserved.

"""Derive laboratory RAG selector opt-ins for non-strict host wiring (FC-3)."""

from __future__ import annotations

from intergrax.applications._shared.wiring_posture import wiring_fails_closed
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.integrations.contracts.base import IntegrationCategory, LabSelectorOptIn
from intergrax.integrations.contracts.integration_profile import IntegrationProfile
from intergrax.rag.embedding.registry.provider_authority import lab_embedding_provider_slug
from intergrax.rag.vectorstore.bootstrap.integration_vectorstore import lab_inmemory_vector_store_slug


def resolve_rag_bootstrap_lab_opt_ins(
    env: ApplicationEnvironmentProfile,
    integration_profile: IntegrationProfile | None = None,
) -> tuple[LabSelectorOptIn | None, LabSelectorOptIn | None]:
    """Return (embedding, vectorstore) opt-ins when laboratory defaults apply."""
    if wiring_fails_closed(env):
        return None, None
    profile = integration_profile or env.integration_profile or IntegrationProfile()
    embedding_opt: LabSelectorOptIn | None = None
    vector_opt: LabSelectorOptIn | None = None
    if profile.slug_for_category(IntegrationCategory.EMBEDDING_PROVIDER) is None:
        embedding_opt = LabSelectorOptIn(
            selector=lab_embedding_provider_slug(),
            reason="derived laboratory embedding default for RAG bootstrap",
        )
    if profile.slug_for_category(IntegrationCategory.VECTOR_STORE) is None:
        vector_opt = LabSelectorOptIn(
            selector=lab_inmemory_vector_store_slug(),
            reason="derived laboratory vector store default for RAG bootstrap",
        )
    return embedding_opt, vector_opt
