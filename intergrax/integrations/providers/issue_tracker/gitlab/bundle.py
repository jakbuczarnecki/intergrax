# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""GitLab integration bundle — composition root for ``"gitlab"``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from intergrax.integrations.contracts.issue_tracker import IssueTracker
from intergrax.integrations.providers.issue_tracker.gitlab.adapter import _GitLabIssueTracker
from intergrax.integrations.providers.issue_tracker.gitlab.client import GitLabRestClient
from intergrax.integrations.providers.issue_tracker.gitlab.config import GitLabIntegrationConfig
from intergrax.integrations.providers.issue_tracker.gitlab.opens import open_gitlab_issue_tracker, open_gitlab_rest_client


@dataclass(frozen=True)
class GitLabIntegrationBundle:
    config: GitLabIntegrationConfig
    issue_tracker: GitlabIssueTrackerIntegration
    rest_client: GitLabRestClient


def resolve_gitlab_config(**overrides: object) -> GitLabIntegrationConfig:
    return GitLabIntegrationConfig.from_env(**overrides)


def create_gitlab_integration(
    *,
    issue_tracker: Optional[IssueTracker] = None,
    client: Optional[GitLabRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[GitLabIntegrationConfig], Any]] = None,
    **config_overrides: object,
) -> GitLabIntegrationBundle:
    config = resolve_gitlab_config(**config_overrides)
    rest_client = client or open_gitlab_rest_client(
        config,
        http_client=http_client,
        http_client_factory=http_client_factory,
    )
    tracker = open_gitlab_issue_tracker(config, implementation=issue_tracker, client=rest_client)
    assert isinstance(tracker, GitLabIssueTracker)
    return GitLabIntegrationBundle(config=config, issue_tracker=tracker, rest_client=rest_client)


def create_gitlab_issue_tracker(
    *,
    issue_tracker: Optional[IssueTracker] = None,
    client: Optional[GitLabRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[GitLabIntegrationConfig], Any]] = None,
    **config_overrides: object,
) -> GitlabIssueTrackerIntegration:
    """Catalog factory for ``"gitlab"`` / ``ISSUE_TRACKER``."""
    return create_gitlab_integration(
        issue_tracker=issue_tracker,
        client=client,
        http_client=http_client,
        http_client_factory=http_client_factory,
        **config_overrides,
    ).issue_tracker

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.issue_tracker.gitlab.integration import (
    GITLAB_ISSUE_TRACKER_PROVIDER_ID,
    GitlabIssueTrackerIntegration,
    GitlabIssueTrackerIntegrationConfig,
    GitlabIssueTrackerClient,
)


def create_gitlab_issue_tracker_integration(
    *,
    client: GitlabIssueTrackerClient | None = None,
    enabled: bool = False,
) -> GitlabIssueTrackerIntegration:
    """
    Build a contract-based Gitlab issue tracker integration.

    Compatibility shim — constructs Integration via from_store (create_gitlab_integration) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Gitlab issue tracker integration requires an injected client when enabled=True",
        )
    if client is not None:
        return GitlabIssueTrackerIntegration.from_client(client, enabled=enabled)
    return GitlabIssueTrackerIntegration.for_provider(
        provider_id=GITLAB_ISSUE_TRACKER_PROVIDER_ID,
        display_name="Gitlab",
        config=GitlabIssueTrackerIntegrationConfig(enabled=enabled),
    )
