# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p5.factories import create_asana_issue_tracker

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.issue_tracker.asana.integration import (
    ASANA_ISSUE_TRACKER_PROVIDER_ID,
    AsanaIssueTrackerIntegration,
    AsanaIssueTrackerIntegrationConfig,
    AsanaIssueTrackerClient,
)

__all__ = [
    "create_asana_issue_tracker",
    "create_asana_issue_tracker_integration",
]


def create_asana_issue_tracker_integration(
    *,
    client: AsanaIssueTrackerClient | None = None,
    enabled: bool = False,
) -> AsanaIssueTrackerIntegration:
    """
    Build a contract-based Asana issue tracker integration.

    The legacy facade (create_asana_issue_tracker) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Asana issue tracker integration requires an injected client when enabled=True",
        )
    if client is not None:
        return AsanaIssueTrackerIntegration.from_client(client, enabled=enabled)
    return AsanaIssueTrackerIntegration.for_provider(
        provider_id=ASANA_ISSUE_TRACKER_PROVIDER_ID,
        display_name="Asana",
        config=AsanaIssueTrackerIntegrationConfig(enabled=enabled),
    )
