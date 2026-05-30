# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Wiki / knowledge base integration contract (§7.1.2, Phase M.6)."""

from __future__ import annotations

from typing import Optional, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field


class WikiPageRecord(BaseModel):
    """Normalized wiki page for RAG ingestion and agent context."""

    id: str
    title: str
    space_key: str = ""
    body: str = ""
    url: str = ""
    version: Optional[int] = None


class WikiSearchResult(BaseModel):
    pages: Sequence[WikiPageRecord] = Field(default_factory=list)
    total: int = 0


@runtime_checkable
class WikiKnowledge(Protocol):
    """
    Backend-agnostic wiki / knowledge base facade.

    Implementations: confluence, notion, sharepoint, …
    """

    def get_page(self, page_id: str) -> WikiPageRecord:
        """Fetch a page by provider id."""

    def search_pages(self, query: str, *, limit: int = 25) -> WikiSearchResult:
        """Search pages using provider-native query (CQL for Confluence)."""
