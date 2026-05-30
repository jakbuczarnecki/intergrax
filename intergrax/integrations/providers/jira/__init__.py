# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Jira issue tracker integration (Phase M.6)."""

from intergrax.integrations.providers.jira.config import (
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
]

_LAZY_EXPORTS = frozenset(
    {
        "JiraIntegrationBundle",
        "JiraIssueTracker",
        "create_jira_integration",
        "create_jira_issue_tracker",
        "register_jira_integration",
        "resolve_jira_config",
    }
)


def __getattr__(name: str):
    if name == "register_jira_integration":
        from intergrax.integrations.providers.jira.register import register_jira_integration

        return register_jira_integration
    if name in _LAZY_EXPORTS:
        from intergrax.integrations.providers.jira import bundle as _bundle

        return getattr(_bundle, name)
    if name == "JiraIssueTracker":
        from intergrax.integrations.providers.jira.adapter import JiraIssueTracker

        return JiraIssueTracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
