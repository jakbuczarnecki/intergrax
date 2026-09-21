# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical structural contract for ``websearch.query`` executor injection."""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from intergrax.websearch.schemas.web_search_result import WebSearchResult


@runtime_checkable
class WebSearchQueryExecutor(Protocol):
    """
    Provider-neutral synchronous web search port for tool wiring.

    Owned by the ``websearch.query`` tool consumer; implemented structurally by
    platform ``WebSearchExecutor`` or host-supplied custom executors.
    """

    def search_sync(
        self,
        query: str,
        top_k: Optional[int] = None,
        locale: Optional[str] = None,
        region: Optional[str] = None,
        language: Optional[str] = None,
        safe_search: Optional[bool] = None,
    ) -> list[WebSearchResult]:
        ...
