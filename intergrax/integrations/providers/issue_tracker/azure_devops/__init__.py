# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "AZURE_DEVOPS_ISSUE_TRACKER_PROVIDER_ID",
    "AzureDevopsIssueTrackerIntegration",
    "AzureDevopsIssueTrackerIntegrationConfig",
    "AzureDevopsIssueTrackerClient",
    "create_azure_devops_issue_tracker",
    "create_azure_devops_issue_tracker_integration",
    "register_azure_devops_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_azure_devops_issue_tracker",
        "create_azure_devops_issue_tracker_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "AZURE_DEVOPS_ISSUE_TRACKER_PROVIDER_ID",
        "AzureDevopsIssueTrackerIntegration",
        "AzureDevopsIssueTrackerIntegrationConfig",
        "AzureDevopsIssueTrackerClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "AZURE_DEVOPS_ISSUE_TRACKER_PROVIDER_ID",
        "AzureDevopsIssueTrackerIntegration",
        "AzureDevopsIssueTrackerIntegrationConfig",
        "AzureDevopsIssueTrackerClient",
    }
)

def __getattr__(name: str):
    if name == "register_azure_devops_integration":
        from intergrax.integrations.providers.issue_tracker.azure_devops.register import register_azure_devops_integration

        return register_azure_devops_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.issue_tracker.azure_devops import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.issue_tracker.azure_devops import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.issue_tracker.azure_devops import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
