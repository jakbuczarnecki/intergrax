# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, Field


class HttpRequestInput(BaseModel):
    method: str = Field(default="GET", description="HTTP method (GET, POST, PUT, PATCH, DELETE).")
    url: str = Field(..., min_length=1)
    headers: dict[str, str] = Field(default_factory=dict)
    body: str = ""
    timeout_s: float = Field(default=30.0, ge=1.0, le=120.0)


class HttpRequestOutput(BaseModel):
    success: bool
    status_code: int = 0
    body: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    error: str = ""
