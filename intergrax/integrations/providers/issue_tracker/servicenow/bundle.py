# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p5.factories import create_servicenow_issue_tracker as _legacy_create_servicenow_issue_tracker

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.issue_tracker.servicenow.integration import (
    SERVICENOW_ISSUE_TRACKER_PROVIDER_ID,
    ServicenowIssueTrackerIntegration,
    ServicenowIssueTrackerIntegrationConfig,
    ServicenowIssueTrackerClient,
)

__all__ = [
    "create_servicenow_issue_tracker",
    "create_servicenow_issue_tracker_integration",
]


def create_servicenow_issue_tracker_integration(
    *,
    client: ServicenowIssueTrackerClient | None = None,
    enabled: bool = False,
) -> ServicenowIssueTrackerIntegration:
    """
    Build a contract-based Servicenow issue tracker integration.

    The legacy facade (create_servicenow_issue_tracker) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Servicenow issue tracker integration requires an injected client when enabled=True",
        )
    if client is not None:
        return ServicenowIssueTrackerIntegration.from_client(client, enabled=enabled)
    return ServicenowIssueTrackerIntegration.for_provider(
        provider_id=SERVICENOW_ISSUE_TRACKER_PROVIDER_ID,
        display_name="Servicenow",
        config=ServicenowIssueTrackerIntegrationConfig(enabled=enabled),
    )


def create_servicenow_issue_tracker(**kwargs: object) -> ServicenowIssueTrackerIntegration:
    """Compatibility shim — constructs ServicenowIssueTrackerIntegration from legacy runtime."""
    runtime = _legacy_create_servicenow_issue_tracker(**kwargs)
    if isinstance(runtime, ServicenowIssueTrackerIntegration):
        return runtime
    return ServicenowIssueTrackerIntegration.from_client(runtime)
