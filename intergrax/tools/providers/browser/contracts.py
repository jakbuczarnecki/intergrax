# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.tools.providers.provider_resource_enforcement import DEFAULT_BROWSER_MAX_TEXT_CHARS


class BrowserFetchPageInput(BaseModel):
    url: str = Field(..., min_length=1)
    wait_until: str = Field(default="load", description="Navigation wait condition (load, domcontentloaded, networkidle).")
    navigation_timeout_s: float = Field(
        default=30.0,
        ge=1.0,
        le=120.0,
        description="Declared navigation budget (integration may enforce).",
    )
    max_text_chars: int = Field(
        default=DEFAULT_BROWSER_MAX_TEXT_CHARS,
        ge=256,
        le=500_000,
        description="Maximum visible text characters retained from the page.",
    )


class BrowserFetchPageOutput(BaseModel):
    url: str
    title: str = ""
    text: str = ""
    status_code: int = 200
    html_length: int = 0
    truncated: bool = False
