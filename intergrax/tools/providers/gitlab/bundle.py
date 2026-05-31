# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.gitlab.contracts import GitLabCreateIssueInput, GitLabCreateIssueOutput
from intergrax.tools.providers.gitlab.handlers import GitLabCreateIssueHandler
from intergrax.tools.providers.gitlab.service import GITLAB_CREATE_ISSUE_TOOL_ID
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

GITLAB_BUNDLE_ID = "gitlab"
GITLAB_TOOL_IDS: tuple[str, ...] = (GITLAB_CREATE_ISSUE_TOOL_ID,)


def register_gitlab_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=GITLAB_CREATE_ISSUE_TOOL_ID,
            name=GITLAB_CREATE_ISSUE_TOOL_ID,
            description="Create a GitLab issue in the configured project.",
            description_short="Create GitLab issue.",
            input_schema=GitLabCreateIssueInput,
            output_schema=GitLabCreateIssueOutput,
            error_mapping={},
            side_effects=True,
            category="issue_tracker",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("gitlab", "issue_tracker"),
        ),
        GitLabCreateIssueHandler(ctx),
    )
