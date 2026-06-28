# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "SERVICENOW_ISSUE_TRACKER_PROVIDER_ID",
    "ServicenowIssueTrackerIntegration",
    "ServicenowIssueTrackerIntegrationConfig",
    "ServicenowIssueTrackerClient",
    "create_servicenow_issue_tracker",
    "create_servicenow_issue_tracker_integration",
    "register_servicenow_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_servicenow_issue_tracker",
        "create_servicenow_issue_tracker_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "SERVICENOW_ISSUE_TRACKER_PROVIDER_ID",
        "ServicenowIssueTrackerIntegration",
        "ServicenowIssueTrackerIntegrationConfig",
        "ServicenowIssueTrackerClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "SERVICENOW_ISSUE_TRACKER_PROVIDER_ID",
        "ServicenowIssueTrackerIntegration",
        "ServicenowIssueTrackerIntegrationConfig",
        "ServicenowIssueTrackerClient",
    }
)

def __getattr__(name: str):
    if name == "register_servicenow_integration":
        from intergrax.integrations.providers.issue_tracker.servicenow.register import register_servicenow_integration

        return register_servicenow_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.issue_tracker.servicenow import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.issue_tracker.servicenow import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.issue_tracker.servicenow import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
