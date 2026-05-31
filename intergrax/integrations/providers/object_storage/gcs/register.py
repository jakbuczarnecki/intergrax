# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register gcs."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.object_storage.gcs.bundle import create_gcs_object_storage
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_gcs_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.GCS.value,
            categories=(IntegrationCategory.OBJECT_STORAGE,),
            factory=create_gcs_object_storage,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_GCS",
            description="gcs integration (Phase M.6 P2/P3)",
        ),
        override=override,
    )
