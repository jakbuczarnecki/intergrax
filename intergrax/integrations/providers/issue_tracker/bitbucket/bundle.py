# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p5.factories import create_bitbucket_issue_tracker as _legacy_create_bitbucket_issue_tracker

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.issue_tracker.bitbucket.integration import (
    BITBUCKET_ISSUE_TRACKER_PROVIDER_ID,
    BitbucketIssueTrackerIntegration,
    BitbucketIssueTrackerIntegrationConfig,
    BitbucketIssueTrackerClient,
)

__all__ = [
    "create_bitbucket_issue_tracker",
    "create_bitbucket_issue_tracker_integration",
]


def create_bitbucket_issue_tracker_integration(
    *,
    client: BitbucketIssueTrackerClient | None = None,
    enabled: bool = False,
) -> BitbucketIssueTrackerIntegration:
    """
    Build a contract-based Bitbucket issue tracker integration.

    The legacy facade (create_bitbucket_issue_tracker) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Bitbucket issue tracker integration requires an injected client when enabled=True",
        )
    if client is not None:
        return BitbucketIssueTrackerIntegration.from_client(client, enabled=enabled)
    return BitbucketIssueTrackerIntegration.for_provider(
        provider_id=BITBUCKET_ISSUE_TRACKER_PROVIDER_ID,
        display_name="Bitbucket",
        config=BitbucketIssueTrackerIntegrationConfig(enabled=enabled),
    )


def create_bitbucket_issue_tracker(**kwargs: object) -> BitbucketIssueTrackerIntegration:
    """Compatibility shim — constructs BitbucketIssueTrackerIntegration from legacy runtime."""
    runtime = _legacy_create_bitbucket_issue_tracker(**kwargs)
    if isinstance(runtime, BitbucketIssueTrackerIntegration):
        return runtime
    return BitbucketIssueTrackerIntegration.from_client(runtime)
