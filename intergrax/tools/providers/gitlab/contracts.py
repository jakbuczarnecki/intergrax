# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class GitLabCreateIssueInput(BaseModel):
    title: str = Field(..., min_length=1)
    description: str = ""
    labels: list[str] = Field(default_factory=list)


class GitLabIssueOutput(BaseModel):
    key: str
    summary: str
    description: str = ""
    status: str = ""
    url: str = ""


class GitLabCreateIssueOutput(BaseModel):
    issue: GitLabIssueOutput
    created: bool = True
