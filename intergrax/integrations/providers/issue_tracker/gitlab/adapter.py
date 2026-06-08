# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""GitLab issue tracker adapter — ``IssueTracker`` facade."""

from __future__ import annotations

from typing import Optional

from intergrax.integrations.contracts.issue_tracker import IssueComment, IssueRecord, IssueSearchResult
from intergrax.integrations.providers.issue_tracker.gitlab.client import GitLabRestClient


class GitLabIssueTracker:
    """Catalog facade over ``GitLabRestClient``."""

    def __init__(self, client: GitLabRestClient) -> None:
        self._client = client

    @property
    def rest_client(self) -> GitLabRestClient:
        return self._client

    def get_issue(self, issue_key: str) -> IssueRecord:
        return self._client.get_issue(issue_key)

    def add_comment(self, issue_key: str, body: str) -> IssueComment:
        return self._client.add_comment(issue_key, body)

    def search_issues(self, jql: str, *, limit: int = 50) -> IssueSearchResult:
        return self._client.search_issues(jql, limit=limit)

    def create_issue(
        self,
        *,
        title: str,
        description: str = "",
        labels: Optional[list[str]] = None,
    ) -> IssueRecord:
        return self._client.create_issue(title=title, description=description, labels=labels)

    def update_issue(
        self,
        issue_key: str,
        *,
        status: str | None = None,
        assignee: str | None = None,
        summary: str | None = None,
    ) -> IssueRecord:
        return self._client.update_issue(
            issue_key,
            status=status,
            assignee=assignee,
            summary=summary,
        )
