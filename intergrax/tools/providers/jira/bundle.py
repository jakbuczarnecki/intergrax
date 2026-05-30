# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.jira.contracts import (
    JiraAddCommentInput,
    JiraCommentOutput,
    JiraGetIssueInput,
    JiraIssueOutput,
    JiraSearchTasksInput,
    JiraSearchTasksOutput,
)
from intergrax.tools.providers.jira.handlers import (
    JiraAddCommentHandler,
    JiraGetIssueHandler,
    JiraSearchTasksHandler,
)
from intergrax.tools.providers.jira.service import (
    JIRA_ADD_COMMENT_TOOL_ID,
    JIRA_GET_ISSUE_TOOL_ID,
    JIRA_SEARCH_TASKS_TOOL_ID,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

JIRA_BUNDLE_ID = "jira"
JIRA_TOOL_IDS: tuple[str, ...] = (
    JIRA_GET_ISSUE_TOOL_ID,
    JIRA_ADD_COMMENT_TOOL_ID,
    JIRA_SEARCH_TASKS_TOOL_ID,
)


def register_jira_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=JIRA_GET_ISSUE_TOOL_ID,
            name=JIRA_GET_ISSUE_TOOL_ID,
            description="Fetch a Jira issue by key (summary, status, assignee, description).",
            description_short="Get Jira issue by key.",
            input_schema=JiraGetIssueInput,
            output_schema=JiraIssueOutput,
            error_mapping={},
            side_effects=False,
            category="issue_tracker",
            risk_level=ToolRiskLevel.LOW,
            tags=("jira", "issue_tracker"),
        ),
        JiraGetIssueHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=JIRA_ADD_COMMENT_TOOL_ID,
            name=JIRA_ADD_COMMENT_TOOL_ID,
            description="Add a comment to an existing Jira issue.",
            description_short="Comment on a Jira issue.",
            input_schema=JiraAddCommentInput,
            output_schema=JiraCommentOutput,
            error_mapping={},
            side_effects=True,
            category="issue_tracker",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("jira", "issue_tracker"),
        ),
        JiraAddCommentHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=JIRA_SEARCH_TASKS_TOOL_ID,
            name=JIRA_SEARCH_TASKS_TOOL_ID,
            description=(
                "Search Jira issues by project, status, and assignee. "
                "Builds JQL internally — do not pass raw JQL."
            ),
            description_short="Search Jira tasks by filters.",
            input_schema=JiraSearchTasksInput,
            output_schema=JiraSearchTasksOutput,
            error_mapping={},
            side_effects=False,
            category="issue_tracker",
            risk_level=ToolRiskLevel.LOW,
            tags=("jira", "issue_tracker"),
        ),
        JiraSearchTasksHandler(ctx),
    )
