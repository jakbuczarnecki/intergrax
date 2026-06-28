# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Jira issue tracker integration (Phase M.6)."""

from intergrax.utils.lazy_export import export_from_bundle
from intergrax.integrations.providers.issue_tracker.jira.config import (
    ENV_JIRA_API_TOKEN,
    ENV_JIRA_BASE_URL,
    ENV_JIRA_EMAIL,
    JiraIntegrationConfig,
)

__all__ = [
    "ENV_JIRA_API_TOKEN",
    "ENV_JIRA_BASE_URL",
    "ENV_JIRA_EMAIL",
    "JiraIntegrationBundle",
    "JiraIntegrationConfig",
    "JiraIssueTracker",
    "create_jira_integration",
    "create_jira_issue_tracker",
    "register_jira_integration",
    "resolve_jira_config",
    "create_jira_issue_tracker_integration",
]

_LAZY_EXPORTS = frozenset(
    {
        "JiraIntegrationBundle",
        "JiraIssueTracker",
        "create_jira_integration",
        "create_jira_issue_tracker",
        "register_jira_integration",
        "resolve_jira_config",
        "create_jira_issue_tracker_integration",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "JIRA_ISSUE_TRACKER_PROVIDER_ID",
        "JiraIssueTrackerIntegration",
        "JiraIssueTrackerIntegrationConfig",
        "JiraIssueTrackerClient",
    }
)

def __getattr__(name: str):
    if name == "register_jira_integration":
        from intergrax.integrations.providers.issue_tracker.jira.register import register_jira_integration

        return register_jira_integration
    if name in _LAZY_EXPORTS:
        from intergrax.integrations.providers.issue_tracker.jira import bundle as _bundle

        return export_from_bundle(_bundle, name, _LAZY_EXPORTS)
    if name == "JiraIssueTracker":
        from intergrax.integrations.providers.issue_tracker.jira.adapter import JiraIssueTracker

        return JiraIssueTracker
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.issue_tracker.jira import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
