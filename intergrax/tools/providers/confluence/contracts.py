# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ConfluenceGetPageInput(BaseModel):
    page_id: str = Field(..., min_length=1, description="Confluence page id.")


class ConfluencePageOutput(BaseModel):
    id: str
    title: str
    space_key: str = ""
    body: str = ""
    url: str = ""
    version: Optional[int] = None


class ConfluenceSearchPagesInput(BaseModel):
    query: str = Field(..., min_length=1, description="Search query for wiki pages.")
    limit: int = Field(default=10, ge=1, le=50)


class ConfluenceSearchPagesOutput(BaseModel):
    pages: list[ConfluencePageOutput] = Field(default_factory=list)
    total: int = 0
