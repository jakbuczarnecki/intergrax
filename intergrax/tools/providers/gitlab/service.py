# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.issue_tracker import IssueCreator, IssueRecord
from intergrax.tools.providers.gitlab.contracts import GitLabCreateIssueInput, GitLabCreateIssueOutput, GitLabIssueOutput
from intergrax.tools.registry.wiring import ToolWiringContext

GITLAB_CREATE_ISSUE_TOOL_ID = "gitlab.create_issue"


def _to_output(record: IssueRecord) -> GitLabIssueOutput:
    return GitLabIssueOutput(
        key=record.key,
        summary=record.summary,
        description=record.description,
        status=record.status,
        url=record.url,
    )


def gitlab_create_issue(ctx: ToolWiringContext, params: GitLabCreateIssueInput) -> GitLabCreateIssueOutput:
    tracker = ctx.issue_tracker
    if tracker is None:
        raise RuntimeError("issue_tracker_not_configured")
    if not isinstance(tracker, IssueCreator):
        raise RuntimeError("issue_tracker_does_not_support_create_issue")
    record = tracker.create_issue(
        title=params.title.strip(),
        description=params.description,
        labels=params.labels or None,
    )
    return GitLabCreateIssueOutput(issue=_to_output(record))
