# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register filesystem."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.object_storage.filesystem.bundle import create_filesystem_object_storage
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_filesystem_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.FILESYSTEM.value,
            categories=(IntegrationCategory.OBJECT_STORAGE,),
            factory=create_filesystem_object_storage,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_FILESYSTEM",
            description="filesystem integration (Phase M.7)",
        ),
        override=override,
    )
