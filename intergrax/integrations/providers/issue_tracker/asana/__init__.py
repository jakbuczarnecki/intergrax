# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "ASANA_ISSUE_TRACKER_PROVIDER_ID",
    "AsanaIssueTrackerIntegration",
    "AsanaIssueTrackerIntegrationConfig",
    "AsanaIssueTrackerClient",
    "create_asana_issue_tracker",
    "create_asana_issue_tracker_integration",
    "register_asana_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_asana_issue_tracker",
        "create_asana_issue_tracker_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "ASANA_ISSUE_TRACKER_PROVIDER_ID",
        "AsanaIssueTrackerIntegration",
        "AsanaIssueTrackerIntegrationConfig",
        "AsanaIssueTrackerClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "ASANA_ISSUE_TRACKER_PROVIDER_ID",
        "AsanaIssueTrackerIntegration",
        "AsanaIssueTrackerIntegrationConfig",
        "AsanaIssueTrackerClient",
    }
)

def __getattr__(name: str):
    if name == "register_asana_integration":
        from intergrax.integrations.providers.issue_tracker.asana.register import register_asana_integration

        return register_asana_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.issue_tracker.asana import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.issue_tracker.asana import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.issue_tracker.asana import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
