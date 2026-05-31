# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register S3 in the integration catalog (Phase M.6 P2)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.object_storage.s3.bundle import create_s3_object_storage
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_s3_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.S3.value,
            categories=(IntegrationCategory.OBJECT_STORAGE,),
            factory=create_s3_object_storage,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_S3",
            description="AWS S3 object storage (put/get/delete/presigned_url)",
        ),
        override=override,
    )
