# © Artur Czarnecki. All rights reserved.

"""FC-3 gate helpers — inmemory ``vector_store`` slug for harness / RAG composition tests."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.integration_profile import IntegrationProfile
from intergrax.integrations.contracts.shipped_manifests import INMEMORY

__all__ = [
    "ensure_inmemory_vector_store_slug",
    "attach_inmemory_vector_store_slug",
]


def ensure_inmemory_vector_store_slug(
    profile: IntegrationProfile | None = None,
) -> IntegrationProfile:
    """Return a profile that satisfies FC-3 vector_store slug fail-closed."""
    base = profile or IntegrationProfile()
    if base.slug_for_category(IntegrationCategory.VECTOR_STORE) is None:
        return base.model_copy(update={"vector_store": INMEMORY})
    return base


def attach_inmemory_vector_store_slug(environment: ApplicationEnvironmentProfile) -> None:
    """Attach the lab inmemory vector store slug when the environment profile omits it."""
    environment.integration_profile = ensure_inmemory_vector_store_slug(environment.integration_profile)
