# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.object_storage.azure_blob.bundle import create_azure_blob_object_storage
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_azure_blob_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.AZURE_BLOB.value,
            categories=(IntegrationCategory.OBJECT_STORAGE,),
            factory=create_azure_blob_object_storage,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_AZURE_BLOB",
            description="Azure Blob Storage object storage (put/get/delete/presigned_url)",
        ),
        override=override,
    )
