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
