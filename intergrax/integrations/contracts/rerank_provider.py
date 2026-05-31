# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Rerank provider integration contract (Phase M.7)."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from langchain_core.documents import Document


@runtime_checkable
class RerankProvider(Protocol):
    """Vendor reranking API (Cohere, Jina, …)."""

    def name(self) -> str:
        """Registry name, e.g. ``cohere``."""

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        *,
        top_n: int | None = None,
    ) -> Sequence[Document]:
        """Return documents reordered by relevance to ``query``."""
