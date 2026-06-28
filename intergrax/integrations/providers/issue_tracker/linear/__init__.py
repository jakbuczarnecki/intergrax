# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "LINEAR_ISSUE_TRACKER_PROVIDER_ID",
    "LinearIssueTrackerIntegration",
    "LinearIssueTrackerIntegrationConfig",
    "LinearIssueTrackerClient",
    "create_linear_issue_tracker",
    "create_linear_issue_tracker_integration",
    "register_linear_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_linear_issue_tracker",
        "create_linear_issue_tracker_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "LINEAR_ISSUE_TRACKER_PROVIDER_ID",
        "LinearIssueTrackerIntegration",
        "LinearIssueTrackerIntegrationConfig",
        "LinearIssueTrackerClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "LINEAR_ISSUE_TRACKER_PROVIDER_ID",
        "LinearIssueTrackerIntegration",
        "LinearIssueTrackerIntegrationConfig",
        "LinearIssueTrackerClient",
    }
)

def __getattr__(name: str):
    if name == "register_linear_integration":
        from intergrax.integrations.providers.issue_tracker.linear.register import register_linear_integration

        return register_linear_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.issue_tracker.linear import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.issue_tracker.linear import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.issue_tracker.linear import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
