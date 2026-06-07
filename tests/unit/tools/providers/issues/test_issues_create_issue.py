# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.integrations.contracts.issue_tracker import IssueComment, IssueRecord, IssueSearchResult
from intergrax.tools.providers.issues.contracts import IssuesCreateIssueInput
from intergrax.tools.providers.issues.service import issues_create_issue
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class _ReadOnlyTracker:
    def get_issue(self, issue_key: str) -> IssueRecord:
        return IssueRecord(key=issue_key, summary="x")

    def add_comment(self, issue_key: str, body: str) -> IssueComment:
        return IssueComment(id="1", body=body)

    def search_issues(self, jql: str, *, limit: int = 50) -> IssueSearchResult:
        return IssueSearchResult()


class _CreatingTracker(_ReadOnlyTracker):
    def create_issue(
        self,
        *,
        title: str,
        description: str = "",
        labels: list[str] | None = None,
    ) -> IssueRecord:
        return IssueRecord(key="NEW-1", summary=title, description=description, url="https://example/issues/1")


def test_issues_create_issue_requires_creator_backend() -> None:
    with pytest.raises(RuntimeError, match="issue_tracker_does_not_support_create_issue"):
        issues_create_issue(
            ToolWiringContext(issue_tracker=_ReadOnlyTracker()),
            IssuesCreateIssueInput(title="Bug"),
        )


def test_issues_create_issue_success() -> None:
    out = issues_create_issue(
        ToolWiringContext(issue_tracker=_CreatingTracker()),
        IssuesCreateIssueInput(title="New task", description="details"),
    )
    assert out.issue.key == "NEW-1"
    assert out.issue.summary == "New task"
