# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "GITHUB_ISSUE_TRACKER_PROVIDER_ID",
    "GithubIssueTrackerIntegration",
    "GithubIssueTrackerIntegrationConfig",
    "GithubIssueTrackerClient",
    "create_github_issue_tracker",
    "create_github_issue_tracker_integration",
    "register_github_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_github_issue_tracker",
        "create_github_issue_tracker_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "GITHUB_ISSUE_TRACKER_PROVIDER_ID",
        "GithubIssueTrackerIntegration",
        "GithubIssueTrackerIntegrationConfig",
        "GithubIssueTrackerClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "GITHUB_ISSUE_TRACKER_PROVIDER_ID",
        "GithubIssueTrackerIntegration",
        "GithubIssueTrackerIntegrationConfig",
        "GithubIssueTrackerClient",
    }
)

def __getattr__(name: str):
    if name == "register_github_integration":
        from intergrax.integrations.providers.issue_tracker.github.register import register_github_integration

        return register_github_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.issue_tracker.github import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.issue_tracker.github import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.issue_tracker.github import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
