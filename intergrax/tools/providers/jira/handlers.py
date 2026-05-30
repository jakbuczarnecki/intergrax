# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.providers.jira.contracts import (
    JiraAddCommentInput,
    JiraCommentOutput,
    JiraGetIssueInput,
    JiraIssueOutput,
    JiraSearchTasksInput,
    JiraSearchTasksOutput,
)
from intergrax.tools.providers.jira.service import jira_add_comment, jira_get_issue, jira_search_tasks
from intergrax.tools.registry.wiring import ToolWiringContext


class JiraGetIssueHandler:
    def __init__(self, ctx: ToolWiringContext) -> None:
        self._ctx = ctx

    def execute(self, request: ToolExecutionRequest[JiraGetIssueInput]) -> JiraIssueOutput:
        return jira_get_issue(self._ctx, request.input)


class JiraAddCommentHandler:
    def __init__(self, ctx: ToolWiringContext) -> None:
        self._ctx = ctx

    def execute(self, request: ToolExecutionRequest[JiraAddCommentInput]) -> JiraCommentOutput:
        return jira_add_comment(self._ctx, request.input)


class JiraSearchTasksHandler:
    def __init__(self, ctx: ToolWiringContext) -> None:
        self._ctx = ctx

    def execute(self, request: ToolExecutionRequest[JiraSearchTasksInput]) -> JiraSearchTasksOutput:
        return jira_search_tasks(self._ctx, request.input)
