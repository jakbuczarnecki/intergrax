# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level Jira openers — internal to the jira integration package.

Only this module may construct ``httpx.Client`` / ``JiraRestClient`` for Jira.
All composition roots use ``bundle.create_jira_*`` or ``profile.resolve(ISSUE_TRACKER)``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.integrations.contracts.issue_tracker import IssueTracker
from intergrax.integrations.providers.issue_tracker.jira.adapter import _JiraIssueTracker
from intergrax.integrations.providers.issue_tracker.jira.integration import JiraIssueTrackerIntegration
from intergrax.integrations.providers.issue_tracker.jira.client import JiraRestClient
from intergrax.integrations.providers.issue_tracker.jira.config import DEFAULT_TIMEOUT_SECONDS, JiraIntegrationConfig


def _create_http_client(config: JiraIntegrationConfig) -> Any:
    import httpx

    timeout = float(config.timeout_seconds or DEFAULT_TIMEOUT_SECONDS)
    return httpx.Client(
        base_url=config.api_base_url,
        auth=(config.email, config.api_token),
        timeout=timeout,
        headers={"Accept": "application/json"},
    )


def open_jira_rest_client(
    config: JiraIntegrationConfig,
    *,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[JiraIntegrationConfig], Any]] = None,
) -> JiraRestClient:
    if http_client is None:
        factory = http_client_factory or _create_http_client
        http_client = factory(config)
    return JiraRestClient(config, http_client=http_client)


def open_jira_issue_tracker(
    config: JiraIntegrationConfig,
    *,
    implementation: Optional[IssueTracker] = None,
    client: Optional[JiraRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[JiraIntegrationConfig], Any]] = None,
) -> IssueTracker:
    if implementation is not None:
        return implementation
    rest_client = client or open_jira_rest_client(
        config,
        http_client=http_client,
        http_client_factory=http_client_factory,
    )
    return JiraIssueTrackerIntegration.from_client(_JiraIssueTracker(rest_client))