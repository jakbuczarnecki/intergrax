# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class JiraGetIssueInput(BaseModel):
    issue_key: str = Field(..., min_length=1, description="Issue key, e.g. PROJ-123.")


class JiraIssueOutput(BaseModel):
    key: str
    summary: str
    description: str = ""
    status: str = ""
    assignee: Optional[str] = None
    url: str = ""


class JiraAddCommentInput(BaseModel):
    issue_key: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1, description="Comment text in plain language.")


class JiraCommentOutput(BaseModel):
    id: str
    body: str
    author: Optional[str] = None
    issue_key: str


class JiraSearchTasksInput(BaseModel):
    project: Optional[str] = Field(default=None, description="Jira project key or name.")
    status: Optional[str] = Field(default=None, description="Issue status name.")
    assignee: Optional[str] = Field(default=None, description="Assignee username or email.")
    limit: int = Field(default=20, ge=1, le=50)


class JiraSearchTasksOutput(BaseModel):
    issues: list[JiraIssueOutput] = Field(default_factory=list)
    total: int = 0
    jql: str = ""
