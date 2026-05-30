# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pydantic contracts for ``websearch.query`` (Phase O.3)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class WebsearchQueryInput(BaseModel):
    """LLM-facing input for live web research."""

    query: str = Field(..., min_length=1, description="Web search query.")
    limit: int = Field(default=8, ge=1, le=20, description="Maximum number of results to return.")
    language: Optional[str] = Field(default=None, description="Optional language hint (e.g. en, pl).")
    locale: Optional[str] = Field(default=None, description="Optional locale hint (e.g. en-US).")
    region: Optional[str] = Field(default=None, description="Optional region hint (e.g. US, PL).")
    safe_search: Optional[bool] = Field(default=None, description="Enable provider safe-search when supported.")


class WebsearchResultItem(BaseModel):
    title: str
    url: str
    snippet: str = ""
    text: str = ""
    domain: str = ""
    rank: int = 0
    provider: str = ""


class WebsearchQueryOutput(BaseModel):
    used: bool
    results: list[WebsearchResultItem] = Field(default_factory=list)
    context_text: str = ""
    reason: str = ""
