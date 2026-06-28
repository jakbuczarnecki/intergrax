# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p2.factories import create_azure_devops_issue_tracker

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.issue_tracker.azure_devops.integration import (
    AZURE_DEVOPS_ISSUE_TRACKER_PROVIDER_ID,
    AzureDevopsIssueTrackerIntegration,
    AzureDevopsIssueTrackerIntegrationConfig,
    AzureDevopsIssueTrackerClient,
)

__all__ = [
    "create_azure_devops_issue_tracker",
    "create_azure_devops_issue_tracker_integration",
]


def create_azure_devops_issue_tracker_integration(
    *,
    client: AzureDevopsIssueTrackerClient | None = None,
    enabled: bool = False,
) -> AzureDevopsIssueTrackerIntegration:
    """
    Build a contract-based Azure Devops issue tracker integration.

    The legacy facade (create_azure_devops_issue_tracker) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Azure Devops issue tracker integration requires an injected client when enabled=True",
        )
    if client is not None:
        return AzureDevopsIssueTrackerIntegration.from_client(client, enabled=enabled)
    return AzureDevopsIssueTrackerIntegration.for_provider(
        provider_id=AZURE_DEVOPS_ISSUE_TRACKER_PROVIDER_ID,
        display_name="Azure Devops",
        config=AzureDevopsIssueTrackerIntegrationConfig(enabled=enabled),
    )
