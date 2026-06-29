# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Jira issue tracker adapter — ``IssueTracker`` facade (no HTTP here)."""

from __future__ import annotations

from intergrax.integrations.contracts.issue_tracker import IssueComment, IssueRecord, IssueSearchResult
from intergrax.integrations.providers.issue_tracker.jira.client import JiraRestClient


class _JiraIssueTracker:
    """
    Catalog facade over ``JiraRestClient``.

    Instantiate via ``create_jira_issue_tracker()`` — not from agent code.
    """

    def __init__(self, client: JiraRestClient) -> None:
        self._client = client

    @property
    def rest_client(self) -> JiraRestClient:
        return self._client

    def get_issue(self, issue_key: str) -> IssueRecord:
        return self._client.get_issue(issue_key)

    def add_comment(self, issue_key: str, body: str) -> IssueComment:
        return self._client.add_comment(issue_key, body)

    def search_issues(self, jql: str, *, limit: int = 50) -> IssueSearchResult:
        return self._client.search_issues(jql, limit=limit)

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
