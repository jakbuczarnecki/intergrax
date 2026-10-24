# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.tools.providers.provider_resource_enforcement import DEFAULT_HTTP_MAX_RESPONSE_BYTES


class HttpRequestInput(BaseModel):
    method: str = Field(default="GET", description="HTTP method (GET, POST, PUT, PATCH, DELETE).")
    url: str = Field(..., min_length=1)
    headers: dict[str, str] = Field(default_factory=dict)
    body: str = ""
    timeout_s: float = Field(default=30.0, ge=1.0, le=120.0)
    max_response_bytes: int = Field(
        default=DEFAULT_HTTP_MAX_RESPONSE_BYTES,
        ge=1,
        le=16_777_216,
        description="Maximum UTF-8 response body retained (truncation when exceeded).",
    )


class HttpRequestOutput(BaseModel):
    success: bool
    status_code: int = 0
    body: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    error: str = ""
    truncated: bool = False
