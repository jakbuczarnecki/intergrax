# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Issue tracker integration contract (§7.1.2, Phase M.6)."""

from __future__ import annotations

from typing import Optional, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field


class IssueRecord(BaseModel):
    """Normalized issue row for agent tools and Tier-3 composition."""

    key: str
    summary: str
    description: str = ""
    status: str = ""
    assignee: Optional[str] = None
    url: str = ""


class IssueComment(BaseModel):
    id: str
    body: str
    author: Optional[str] = None


class IssueSearchResult(BaseModel):
    issues: Sequence[IssueRecord] = Field(default_factory=list)
    total: int = 0


@runtime_checkable
class IssueTracker(Protocol):
    """
    Backend-agnostic issue tracker facade.

    Implementations: jira, azure_devops, github, linear, …
    """

    def get_issue(self, issue_key: str) -> IssueRecord:
        """Fetch a single issue by key (e.g. ``PROJ-123``)."""

    def add_comment(self, issue_key: str, body: str) -> IssueComment:
        """Append a comment to an issue."""

    def search_issues(self, jql: str, *, limit: int = 50) -> IssueSearchResult:
        """Search issues using provider-native query language (JQL for Jira)."""
