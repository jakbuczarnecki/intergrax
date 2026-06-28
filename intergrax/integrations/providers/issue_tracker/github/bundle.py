# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p2.factories import create_github_issue_tracker

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.issue_tracker.github.integration import (
    GITHUB_ISSUE_TRACKER_PROVIDER_ID,
    GithubIssueTrackerIntegration,
    GithubIssueTrackerIntegrationConfig,
    GithubIssueTrackerClient,
)

__all__ = [
    "create_github_issue_tracker",
    "create_github_issue_tracker_integration",
]


def create_github_issue_tracker_integration(
    *,
    client: GithubIssueTrackerClient | None = None,
    enabled: bool = False,
) -> GithubIssueTrackerIntegration:
    """
    Build a contract-based Github issue tracker integration.

    The legacy facade (create_github_issue_tracker) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Github issue tracker integration requires an injected client when enabled=True",
        )
    if client is not None:
        return GithubIssueTrackerIntegration.from_client(client, enabled=enabled)
    return GithubIssueTrackerIntegration.for_provider(
        provider_id=GITHUB_ISSUE_TRACKER_PROVIDER_ID,
        display_name="Github",
        config=GithubIssueTrackerIntegrationConfig(enabled=enabled),
    )
