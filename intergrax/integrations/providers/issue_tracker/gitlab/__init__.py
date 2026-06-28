# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "GITLAB_ISSUE_TRACKER_PROVIDER_ID",
    "GitlabIssueTrackerIntegration",
    "GitlabIssueTrackerIntegrationConfig",
    "GitlabIssueTrackerClient",
    "create_gitlab_integration",
    "create_gitlab_issue_tracker",
    "create_gitlab_issue_tracker_integration",
    "register_gitlab_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_gitlab_integration",
        "create_gitlab_issue_tracker",
        "create_gitlab_issue_tracker_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "GITLAB_ISSUE_TRACKER_PROVIDER_ID",
        "GitlabIssueTrackerIntegration",
        "GitlabIssueTrackerIntegrationConfig",
        "GitlabIssueTrackerClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "GITLAB_ISSUE_TRACKER_PROVIDER_ID",
        "GitlabIssueTrackerIntegration",
        "GitlabIssueTrackerIntegrationConfig",
        "GitlabIssueTrackerClient",
    }
)

def __getattr__(name: str):
    if name == "register_gitlab_integration":
        from intergrax.integrations.providers.issue_tracker.gitlab.register import register_gitlab_integration

        return register_gitlab_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.issue_tracker.gitlab import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.issue_tracker.gitlab import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.issue_tracker.gitlab import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
