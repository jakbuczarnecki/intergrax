# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import List, Optional, Sequence

from langchain_core.documents import Document

from intergrax.integrations.providers.rerank_provider.cohere_rerank.config import CohereRerankIntegrationConfig
from intergrax.integrations.providers.rerank_provider.cohere_rerank.opens import cohere_rerank_scores


class _CohereRerankProvider:
    def __init__(self, config: CohereRerankIntegrationConfig) -> None:
        self._config = config

    def name(self) -> str:
        return "cohere"

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        *,
        top_n: int | None = None,
    ) -> Sequence[Document]:
        texts = [d.page_content or "" for d in documents]
        scores = cohere_rerank_scores(self._config, query, texts, top_n=top_n)
        ordered = sorted(
            zip(documents, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        if top_n is not None:
            ordered = ordered[:top_n]
        return [doc for doc, _ in ordered]

    def score(self, query: str, texts: List[str], *, top_n: Optional[int] = None) -> List[float]:
        return cohere_rerank_scores(self._config, query, texts, top_n=top_n)
