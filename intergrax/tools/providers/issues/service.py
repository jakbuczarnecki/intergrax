# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.issue_tracker import IssueRecord, IssueTracker
from intergrax.tools.providers.issues.contracts import (
    IssuesAddCommentInput,
    IssuesCommentOutput,
    IssuesGetIssueInput,
    IssuesIssueOutput,
    IssuesSearchInput,
    IssuesSearchOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

ISSUES_GET_ISSUE_TOOL_ID = "issues.get_issue"
ISSUES_ADD_COMMENT_TOOL_ID = "issues.add_comment"
ISSUES_SEARCH_TOOL_ID = "issues.search"


def _require_tracker(ctx: ToolWiringContext) -> IssueTracker:
    tracker = ctx.issue_tracker
    if tracker is None:
        raise RuntimeError("issue_tracker_not_configured")
    return tracker


def _to_issue_output(record: IssueRecord) -> IssuesIssueOutput:
    return IssuesIssueOutput(
        key=record.key,
        summary=record.summary,
        description=record.description,
        status=record.status,
        assignee=record.assignee,
        url=record.url,
    )


def issues_get_issue(ctx: ToolWiringContext, params: IssuesGetIssueInput) -> IssuesIssueOutput:
    return _to_issue_output(_require_tracker(ctx).get_issue(params.issue_key.strip()))


def issues_add_comment(ctx: ToolWiringContext, params: IssuesAddCommentInput) -> IssuesCommentOutput:
    comment = _require_tracker(ctx).add_comment(params.issue_key.strip(), params.body)
    return IssuesCommentOutput(id=comment.id, body=comment.body, author=comment.author)


def issues_search(ctx: ToolWiringContext, params: IssuesSearchInput) -> IssuesSearchOutput:
    result = _require_tracker(ctx).search_issues(params.query.strip(), limit=params.limit)
    issues = [_to_issue_output(item) for item in result.issues]
    return IssuesSearchOutput(issues=issues, total=int(result.total))
