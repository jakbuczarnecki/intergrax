# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p7.factories import create_zendesk_issue_tracker

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.issue_tracker.zendesk.integration import (
    ZENDESK_ISSUE_TRACKER_PROVIDER_ID,
    ZendeskIssueTrackerIntegration,
    ZendeskIssueTrackerIntegrationConfig,
    ZendeskIssueTrackerClient,
)

__all__ = [
    "create_zendesk_issue_tracker",
    "create_zendesk_issue_tracker_integration",
]


def create_zendesk_issue_tracker_integration(
    *,
    client: ZendeskIssueTrackerClient | None = None,
    enabled: bool = False,
) -> ZendeskIssueTrackerIntegration:
    """
    Build a contract-based Zendesk issue tracker integration.

    The legacy facade (create_zendesk_issue_tracker) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Zendesk issue tracker integration requires an injected client when enabled=True",
        )
    if client is not None:
        return ZendeskIssueTrackerIntegration.from_client(client, enabled=enabled)
    return ZendeskIssueTrackerIntegration.for_provider(
        provider_id=ZENDESK_ISSUE_TRACKER_PROVIDER_ID,
        display_name="Zendesk",
        config=ZendeskIssueTrackerIntegrationConfig(enabled=enabled),
    )
