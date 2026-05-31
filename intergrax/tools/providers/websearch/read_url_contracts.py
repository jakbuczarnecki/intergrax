# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Contracts for ``websearch.read_url``."""

from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl


class WebsearchReadUrlInput(BaseModel):
    url: HttpUrl = Field(..., description="HTTP(S) URL to fetch and extract.")
    timeout_seconds: int = Field(default=20, ge=5, le=60)
    use_advanced_extraction: bool = Field(
        default=False,
        description="Use trafilatura when available for cleaner article text.",
    )


class WebsearchReadUrlOutput(BaseModel):
    used: bool
    url: str = ""
    final_url: str = ""
    title: str = ""
    text: str = ""
    status_code: int | None = None
    reason: str = ""
