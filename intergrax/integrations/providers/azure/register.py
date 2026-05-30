# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register Azure in the integration catalog (Phase M.6)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.azure.bundle import create_azure_cloud_platform
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_azure_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.AZURE.value,
            categories=(IntegrationCategory.CLOUD_PLATFORM,),
            factory=create_azure_cloud_platform,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_AZURE",
            description=(
                "Azure cloud platform facade (MI / service principal; defaults for Blob, Service Bus, Azure SQL)"
            ),
        ),
        override=override,
    )
