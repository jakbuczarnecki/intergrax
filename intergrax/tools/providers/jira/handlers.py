# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.jira.contracts import (
    JiraAddCommentInput,
    JiraCommentOutput,
    JiraGetIssueInput,
    JiraIssueOutput,
    JiraSearchTasksInput,
    JiraSearchTasksOutput,
)
from intergrax.tools.providers.jira.service import jira_add_comment, jira_get_issue, jira_search_tasks


class JiraGetIssueHandler(ServiceToolHandler[JiraGetIssueInput, JiraIssueOutput]):
    _service = jira_get_issue


class JiraAddCommentHandler(ServiceToolHandler[JiraAddCommentInput, JiraCommentOutput]):
    _service = jira_add_comment


class JiraSearchTasksHandler(ServiceToolHandler[JiraSearchTasksInput, JiraSearchTasksOutput]):
    _service = jira_search_tasks
