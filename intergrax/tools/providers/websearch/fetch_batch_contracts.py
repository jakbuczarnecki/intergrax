# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class WebsearchFetchBatchInput(BaseModel):
    urls: list[HttpUrl] = Field(..., min_length=1, max_length=10, description="HTTP(S) URLs to fetch.")
    timeout_seconds: float = Field(default=15.0, ge=1.0, le=60.0)
    use_advanced_extraction: bool = Field(default=False)


class WebsearchFetchBatchPageOutput(BaseModel):
    url: str
    final_url: Optional[str] = None
    title: str = ""
    text: str = ""
    status_code: Optional[int] = None
    used: bool = False
    reason: str = ""


class WebsearchFetchBatchOutput(BaseModel):
    pages: list[WebsearchFetchBatchPageOutput] = Field(default_factory=list)
    success_count: int = 0
    context_text: str = ""
