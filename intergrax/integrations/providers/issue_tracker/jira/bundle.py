# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete Jira integration bundle — the single composition root for Jira in Intergrax.

HTTP clients are opened only in ``opens.py``. Tier-3 code MUST use
``create_jira_issue_tracker()``, ``create_jira_integration()``, or
``profile.resolve(IntegrationCategory.ISSUE_TRACKER)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from intergrax.integrations.contracts.issue_tracker import IssueTracker
from intergrax.integrations.providers.issue_tracker.jira.adapter import JiraIssueTracker
from intergrax.integrations.providers.issue_tracker.jira.client import JiraRestClient
from intergrax.integrations.providers.issue_tracker.jira.config import JiraIntegrationConfig
from intergrax.integrations.providers.issue_tracker.jira.opens import open_jira_issue_tracker, open_jira_rest_client


@dataclass(frozen=True)
class JiraIntegrationBundle:
    config: JiraIntegrationConfig
    issue_tracker: JiraIssueTracker
    rest_client: JiraRestClient


def resolve_jira_config(**overrides: object) -> JiraIntegrationConfig:
    return JiraIntegrationConfig.from_env(**overrides)


def create_jira_integration(
    *,
    issue_tracker: Optional[IssueTracker] = None,
    client: Optional[JiraRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[JiraIntegrationConfig], Any]] = None,
    **config_overrides: object,
) -> JiraIntegrationBundle:
    config = resolve_jira_config(**config_overrides)
    rest_client = client or open_jira_rest_client(
        config,
        http_client=http_client,
        http_client_factory=http_client_factory,
    )
    tracker = open_jira_issue_tracker(config, implementation=issue_tracker, client=rest_client)
    assert isinstance(tracker, JiraIssueTracker)
    return JiraIntegrationBundle(config=config, issue_tracker=tracker, rest_client=rest_client)


def create_jira_issue_tracker(
    *,
    issue_tracker: Optional[IssueTracker] = None,
    client: Optional[JiraRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[JiraIntegrationConfig], Any]] = None,
    **config_overrides: object,
) -> JiraIssueTracker:
    """Catalog factory for ``"jira"`` / ``ISSUE_TRACKER``."""
    return create_jira_integration(
        issue_tracker=issue_tracker,
        client=client,
        http_client=http_client,
        http_client_factory=http_client_factory,
        **config_overrides,
    ).issue_tracker
