# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Security scanner tool I/O contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SecurityScanInput(BaseModel):
    target: str = Field(description="Image reference or repository path to scan.")
    scan_type: Literal["image", "repo"] = Field(default="repo")


class SecurityFindingOutput(BaseModel):
    id: str = ""
    severity: str = ""
    title: str = ""
    resource: str = ""
    detail: str = ""


class SecurityScanOutput(BaseModel):
    target: str
    status: str = "completed"
    findings: list[SecurityFindingOutput] = Field(default_factory=list)


class SecuritySummarizeFindingsInput(BaseModel):
    findings: list[SecurityFindingOutput] = Field(default_factory=list)


class SecuritySeverityCountOutput(BaseModel):
    severity: str
    count: int = 0


class SecuritySummarizeFindingsOutput(BaseModel):
    total: int = 0
    by_severity: list[SecuritySeverityCountOutput] = Field(default_factory=list)
    critical_count: int = 0
    high_count: int = 0
