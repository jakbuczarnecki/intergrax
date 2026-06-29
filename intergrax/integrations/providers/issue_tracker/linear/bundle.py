# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p2.factories import create_linear_issue_tracker as _legacy_create_linear_issue_tracker

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.issue_tracker.linear.integration import (
    LINEAR_ISSUE_TRACKER_PROVIDER_ID,
    LinearIssueTrackerIntegration,
    LinearIssueTrackerIntegrationConfig,
    LinearIssueTrackerClient,
)

__all__ = [
    "create_linear_issue_tracker",
    "create_linear_issue_tracker_integration",
]


def create_linear_issue_tracker_integration(
    *,
    client: LinearIssueTrackerClient | None = None,
    enabled: bool = False,
) -> LinearIssueTrackerIntegration:
    """
    Build a contract-based Linear issue tracker integration.

    The legacy facade (create_linear_issue_tracker) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Linear issue tracker integration requires an injected client when enabled=True",
        )
    if client is not None:
        return LinearIssueTrackerIntegration.from_client(client, enabled=enabled)
    return LinearIssueTrackerIntegration.for_provider(
        provider_id=LINEAR_ISSUE_TRACKER_PROVIDER_ID,
        display_name="Linear",
        config=LinearIssueTrackerIntegrationConfig(enabled=enabled),
    )


def create_linear_issue_tracker(**kwargs: object) -> LinearIssueTrackerIntegration:
    """Compatibility shim — constructs LinearIssueTrackerIntegration from legacy runtime."""
    runtime = _legacy_create_linear_issue_tracker(**kwargs)
    if isinstance(runtime, LinearIssueTrackerIntegration):
        return runtime
    return LinearIssueTrackerIntegration.from_client(runtime)
