# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, Field


class KnowledgeGetPageInput(BaseModel):
    page_id: str = Field(..., min_length=1, description="Provider-native page identifier.")


class KnowledgePageOutput(BaseModel):
    id: str
    title: str
    space_key: str = ""
    body: str = ""
    url: str = ""
    version: int | None = None


class KnowledgeSearchInput(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(default=25, ge=1, le=100)


class KnowledgeSearchOutput(BaseModel):
    pages: list[KnowledgePageOutput] = Field(default_factory=list)
    total: int = 0
