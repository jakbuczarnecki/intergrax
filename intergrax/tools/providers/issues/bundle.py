# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.issues.contracts import (
    IssuesAddCommentInput,
    IssuesCommentOutput,
    IssuesCreateIssueInput,
    IssuesCreateIssueOutput,
    IssuesGetIssueInput,
    IssuesIssueOutput,
    IssuesSearchInput,
    IssuesSearchOutput,
)
from intergrax.tools.providers.issues.handlers import (
    IssuesAddCommentHandler,
    IssuesCreateIssueHandler,
    IssuesGetIssueHandler,
    IssuesSearchHandler,
)
from intergrax.tools.providers.issues.service import (
    ISSUES_ADD_COMMENT_TOOL_ID,
    ISSUES_CREATE_ISSUE_TOOL_ID,
    ISSUES_GET_ISSUE_TOOL_ID,
    ISSUES_SEARCH_TOOL_ID,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

ISSUES_BUNDLE_ID = "issues"
ISSUES_TOOL_IDS: tuple[str, ...] = (
    ISSUES_GET_ISSUE_TOOL_ID,
    ISSUES_ADD_COMMENT_TOOL_ID,
    ISSUES_SEARCH_TOOL_ID,
    ISSUES_CREATE_ISSUE_TOOL_ID,
)


def register_issues_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=ISSUES_GET_ISSUE_TOOL_ID,
            name=ISSUES_GET_ISSUE_TOOL_ID,
            description="Fetch a single issue by key from the configured issue tracker (provider-agnostic).",
            description_short="Get issue.",
            input_schema=IssuesGetIssueInput,
            output_schema=IssuesIssueOutput,
            error_mapping={},
            side_effects=False,
            category="issues",
            risk_level=ToolRiskLevel.LOW,
            tags=("issues", "tracker"),
        ),
        IssuesGetIssueHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=ISSUES_ADD_COMMENT_TOOL_ID,
            name=ISSUES_ADD_COMMENT_TOOL_ID,
            description="Add a comment to an issue in the configured issue tracker.",
            description_short="Comment on issue.",
            input_schema=IssuesAddCommentInput,
            output_schema=IssuesCommentOutput,
            error_mapping={},
            side_effects=True,
            category="issues",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("issues", "tracker"),
        ),
        IssuesAddCommentHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=ISSUES_SEARCH_TOOL_ID,
            name=ISSUES_SEARCH_TOOL_ID,
            description="Search issues using provider-native query language (JQL for Jira, etc.).",
            description_short="Search issues.",
            input_schema=IssuesSearchInput,
            output_schema=IssuesSearchOutput,
            error_mapping={},
            side_effects=False,
            category="issues",
            risk_level=ToolRiskLevel.LOW,
            tags=("issues", "tracker"),
        ),
        IssuesSearchHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=ISSUES_CREATE_ISSUE_TOOL_ID,
            name=ISSUES_CREATE_ISSUE_TOOL_ID,
            description="Create an issue when the configured tracker implements IssueCreator.",
            description_short="Create issue.",
            input_schema=IssuesCreateIssueInput,
            output_schema=IssuesCreateIssueOutput,
            error_mapping={},
            side_effects=True,
            category="issues",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("issues", "tracker"),
        ),
        IssuesCreateIssueHandler(ctx),
    )
