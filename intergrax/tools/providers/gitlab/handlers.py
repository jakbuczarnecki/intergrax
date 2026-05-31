# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.gitlab.contracts import GitLabCreateIssueInput, GitLabCreateIssueOutput
from intergrax.tools.providers.gitlab.service import gitlab_create_issue


class GitLabCreateIssueHandler(ServiceToolHandler[GitLabCreateIssueInput, GitLabCreateIssueOutput]):
    _service = gitlab_create_issue
