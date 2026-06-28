# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "ZENDESK_ISSUE_TRACKER_PROVIDER_ID",
    "ZendeskIssueTrackerIntegration",
    "ZendeskIssueTrackerIntegrationConfig",
    "ZendeskIssueTrackerClient",
    "create_zendesk_issue_tracker",
    "create_zendesk_issue_tracker_integration",
    "register_zendesk_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_zendesk_issue_tracker",
        "create_zendesk_issue_tracker_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "ZENDESK_ISSUE_TRACKER_PROVIDER_ID",
        "ZendeskIssueTrackerIntegration",
        "ZendeskIssueTrackerIntegrationConfig",
        "ZendeskIssueTrackerClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "ZENDESK_ISSUE_TRACKER_PROVIDER_ID",
        "ZendeskIssueTrackerIntegration",
        "ZendeskIssueTrackerIntegrationConfig",
        "ZendeskIssueTrackerClient",
    }
)

def __getattr__(name: str):
    if name == "register_zendesk_integration":
        from intergrax.integrations.providers.issue_tracker.zendesk.register import register_zendesk_integration

        return register_zendesk_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.issue_tracker.zendesk import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.issue_tracker.zendesk import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.issue_tracker.zendesk import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
