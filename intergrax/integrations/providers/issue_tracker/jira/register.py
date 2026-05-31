# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register Jira in the integration catalog (Phase M.6)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.issue_tracker.jira.bundle import create_jira_issue_tracker
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_jira_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.JIRA.value,
            categories=(IntegrationCategory.ISSUE_TRACKER,),
            factory=create_jira_issue_tracker,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_JIRA",
            description="Jira Cloud issue tracker (get_issue, add_comment, search_issues via REST v3)",
        ),
        override=override,
    )
