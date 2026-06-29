# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import List, Sequence

from langchain_core.documents import Document

from intergrax.integrations.providers.rerank_provider.jina_rerank.config import JinaRerankIntegrationConfig
from intergrax.integrations.providers.rerank_provider.jina_rerank.opens import jina_rerank_scores


class _JinaRerankProvider:
    def __init__(self, config: JinaRerankIntegrationConfig) -> None:
        self._config = config

    def name(self) -> str:
        return "jina"

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        *,
        top_n: int | None = None,
    ) -> Sequence[Document]:
        texts = [d.page_content or "" for d in documents]
        scores = jina_rerank_scores(self._config, query, texts)
        ordered = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)
        if top_n is not None:
            ordered = ordered[:top_n]
        return [doc for doc, _ in ordered]

    def score(self, query: str, texts: List[str]) -> List[float]:
        return jina_rerank_scores(self._config, query, texts)
