# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register MongoDB in the integration catalog (Phase M.6 P2)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.mongodb.bundle import create_mongodb_document_store
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_mongodb_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.MONGODB.value,
            categories=(IntegrationCategory.DOCUMENT_STORE,),
            factory=create_mongodb_document_store,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_MONGODB",
            description=(
                "MongoDB flexible document store (partition-scoped get/put/delete/query)"
            ),
        ),
        override=override,
    )
