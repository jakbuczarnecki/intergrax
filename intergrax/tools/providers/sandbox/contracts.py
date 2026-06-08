# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SandboxExecInput(BaseModel):
    operation: str = Field(..., min_length=1, description="Allowlisted sandbox operation name.")
    payload: dict[str, Any] = Field(default_factory=dict, description="Operation-specific payload.")


class SandboxExecOutput(BaseModel):
    success: bool
    output: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    session_id: str = ""


class CodeExecInput(BaseModel):
    code: str = Field(..., min_length=1)
    language: str = Field(default="python", description="Supported: python.")
    timeout_s: int = Field(default=30, ge=1, le=120)


class ScriptRunInput(BaseModel):
    path: str = Field(..., min_length=1, description="Script path relative to sandbox root.")
    args: list[str] = Field(default_factory=list)
    interpreter: str = Field(default="", description="Optional interpreter path; default python.")
    timeout_s: int = Field(default=60, ge=1, le=300)


class BrowserRunInput(BaseModel):
    url: str = Field(..., min_length=1)
    wait_until: str = Field(default="load")
    max_chars: int = Field(default=50_000, ge=256, le=200_000)
    timeout_s: int = Field(default=30, ge=1, le=120)


class BrowserRunOutput(BaseModel):
    success: bool
    url: str = ""
    title: str = ""
    content: str = ""
    error: str = ""
    session_id: str = ""


class SandboxListOperationsInput(BaseModel):
    pass


class SandboxListOperationsOutput(BaseModel):
    session_id: str = ""
    operations: list[str] = Field(default_factory=list)
