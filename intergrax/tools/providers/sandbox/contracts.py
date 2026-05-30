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
