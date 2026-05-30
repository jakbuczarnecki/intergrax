# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.issue_tracker import IssueRecord
from intergrax.tools.providers.jira.contracts import (
    JiraAddCommentInput,
    JiraCommentOutput,
    JiraGetIssueInput,
    JiraIssueOutput,
    JiraSearchTasksInput,
    JiraSearchTasksOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

JIRA_GET_ISSUE_TOOL_ID = "jira.get_issue"
JIRA_ADD_COMMENT_TOOL_ID = "jira.add_comment"
JIRA_SEARCH_TASKS_TOOL_ID = "jira.search_tasks"


def _require_tracker(ctx: ToolWiringContext):
    tracker = ctx.issue_tracker
    if tracker is None:
        raise RuntimeError("issue_tracker_not_configured")
    return tracker


def _to_issue_output(record: IssueRecord) -> JiraIssueOutput:
    return JiraIssueOutput(
        key=record.key,
        summary=record.summary,
        description=record.description,
        status=record.status,
        assignee=record.assignee,
        url=record.url,
    )


def build_jira_jql(params: JiraSearchTasksInput) -> str:
    clauses: list[str] = []
    if params.project:
        clauses.append(f'project = "{params.project}"')
    if params.status:
        clauses.append(f'status = "{params.status}"')
    if params.assignee:
        clauses.append(f'assignee = "{params.assignee}"')
    if not clauses:
        return "order by updated DESC"
    return " AND ".join(clauses) + " order by updated DESC"


def jira_get_issue(ctx: ToolWiringContext, params: JiraGetIssueInput) -> JiraIssueOutput:
    record = _require_tracker(ctx).get_issue(params.issue_key.strip())
    return _to_issue_output(record)


def jira_add_comment(ctx: ToolWiringContext, params: JiraAddCommentInput) -> JiraCommentOutput:
    comment = _require_tracker(ctx).add_comment(params.issue_key.strip(), params.body)
    return JiraCommentOutput(
        id=comment.id,
        body=comment.body,
        author=comment.author,
        issue_key=params.issue_key.strip(),
    )


def jira_search_tasks(ctx: ToolWiringContext, params: JiraSearchTasksInput) -> JiraSearchTasksOutput:
    jql = build_jira_jql(params)
    result = _require_tracker(ctx).search_issues(jql, limit=params.limit)
    issues = [_to_issue_output(issue) for issue in result.issues]
    return JiraSearchTasksOutput(issues=issues, total=int(result.total), jql=jql)
