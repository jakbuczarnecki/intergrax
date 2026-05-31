# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Low-level GitLab openers — httpx only here."""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.integrations.contracts.issue_tracker import IssueTracker
from intergrax.integrations.providers.issue_tracker.gitlab.adapter import GitLabIssueTracker
from intergrax.integrations.providers.issue_tracker.gitlab.client import GitLabRestClient
from intergrax.integrations.providers.issue_tracker.gitlab.config import DEFAULT_TIMEOUT_SECONDS, GitLabIntegrationConfig


def _create_http_client(config: GitLabIntegrationConfig) -> Any:
    import httpx

    timeout = float(config.timeout_seconds or DEFAULT_TIMEOUT_SECONDS)
    return httpx.Client(
        base_url=config.api_base_url,
        headers={"PRIVATE-TOKEN": config.token, "Accept": "application/json"},
        timeout=timeout,
    )


def open_gitlab_rest_client(
    config: GitLabIntegrationConfig,
    *,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[GitLabIntegrationConfig], Any]] = None,
) -> GitLabRestClient:
    if http_client is None:
        factory = http_client_factory or _create_http_client
        http_client = factory(config)
    return GitLabRestClient(config, http_client=http_client)


def open_gitlab_issue_tracker(
    config: GitLabIntegrationConfig,
    *,
    implementation: Optional[IssueTracker] = None,
    client: Optional[GitLabRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[GitLabIntegrationConfig], Any]] = None,
) -> IssueTracker:
    if implementation is not None:
        return implementation
    rest_client = client or open_gitlab_rest_client(
        config,
        http_client=http_client,
        http_client_factory=http_client_factory,
    )
    return GitLabIssueTracker(rest_client)
