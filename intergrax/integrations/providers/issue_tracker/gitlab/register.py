# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register gitlab."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.issue_tracker.gitlab.bundle import create_gitlab_issue_tracker
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_gitlab_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.GITLAB.value,
            categories=(IntegrationCategory.ISSUE_TRACKER,),
            factory=create_gitlab_issue_tracker,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_GITLAB",
            description="gitlab integration (Phase M.8 harness)",
        ),
        override=override,
    )
