# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register github."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.issue_tracker.github.bundle import create_github_issue_tracker
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_github_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.GITHUB.value,
            categories=(IntegrationCategory.ISSUE_TRACKER,),
            factory=create_github_issue_tracker,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_GITHUB",
            description="github integration (Phase M.6 P2/P3)",
        ),
        override=override,
    )
