# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.issues.contracts import (
    IssuesAddCommentInput,
    IssuesCommentOutput,
    IssuesGetIssueInput,
    IssuesIssueOutput,
    IssuesSearchInput,
    IssuesSearchOutput,
)
from intergrax.tools.providers.issues.service import issues_add_comment, issues_get_issue, issues_search


class IssuesGetIssueHandler(ServiceToolHandler[IssuesGetIssueInput, IssuesIssueOutput]):
    _service = issues_get_issue


class IssuesAddCommentHandler(ServiceToolHandler[IssuesAddCommentInput, IssuesCommentOutput]):
    _service = issues_add_comment


class IssuesSearchHandler(ServiceToolHandler[IssuesSearchInput, IssuesSearchOutput]):
    _service = issues_search
