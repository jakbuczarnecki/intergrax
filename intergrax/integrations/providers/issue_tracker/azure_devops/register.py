# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register azure_devops."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.issue_tracker.azure_devops.bundle import create_azure_devops_issue_tracker
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_azure_devops_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.AZURE_DEVOPS.value,
            categories=(IntegrationCategory.ISSUE_TRACKER,),
            factory=create_azure_devops_issue_tracker,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_AZURE_DEVOPS",
            description="azure_devops integration (Phase M.6 P2/P3)",
        ),
        override=override,
    )
