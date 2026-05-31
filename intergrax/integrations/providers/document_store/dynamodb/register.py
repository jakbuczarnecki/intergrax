# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register dynamodb."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.document_store.dynamodb.bundle import create_dynamodb_document_store
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_dynamodb_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.DYNAMODB.value,
            categories=(IntegrationCategory.DOCUMENT_STORE,),
            factory=create_dynamodb_document_store,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_DYNAMODB",
            description="dynamodb integration (Phase M.6 P2/P3)",
        ),
        override=override,
    )
