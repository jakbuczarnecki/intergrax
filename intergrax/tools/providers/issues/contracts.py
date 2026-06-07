# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, Field


class IssuesGetIssueInput(BaseModel):
    issue_key: str = Field(..., min_length=1)


class IssuesIssueOutput(BaseModel):
    key: str
    summary: str
    description: str = ""
    status: str = ""
    assignee: str | None = None
    url: str = ""


class IssuesAddCommentInput(BaseModel):
    issue_key: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)


class IssuesCommentOutput(BaseModel):
    id: str
    body: str
    author: str | None = None


class IssuesSearchInput(BaseModel):
    query: str = Field(..., min_length=1, description="Provider-native query (JQL, etc.).")
    limit: int = Field(default=50, ge=1, le=200)


class IssuesSearchOutput(BaseModel):
    issues: list[IssuesIssueOutput] = Field(default_factory=list)
    total: int = 0
