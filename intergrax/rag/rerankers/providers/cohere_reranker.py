# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import os
from typing import List, Optional

import cohere

from intergrax.rag.rerankers.providers._api_reranker_base import _APIRerankerBase


class CohereReranker(_APIRerankerBase):

    DEFAULT_MODEL = "rerank-english-v3.0"

    ENV_MODEL = "INTERGRAX_DEFAULT_COHERE_RERANK_MODEL"
    ENV_API_KEY = "COHERE_API_KEY"

    def __init__(
        self,
        *,
        client: Optional[cohere.Client] = None,
        model: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> None:

        env_model = os.getenv(self.ENV_MODEL)
        resolved_model = model or env_model or self.DEFAULT_MODEL

        # Lazy initialization fields
        self._client: Optional[cohere.Client] = client
        self._model = resolved_model
        self._top_n = top_n

    @classmethod
    def name(cls) -> str:
        return "cohere"

    def _ensure_client(self) -> None:

        if self._client is not None:
            return

        api_key = os.getenv(self.ENV_API_KEY)

        if not api_key:
            raise RuntimeError(
                "COHERE_API_KEY not found in environment variables."
            )

        self._client = cohere.Client(api_key)

    def _score(
        self,
        query: str,
        texts: List[str],
    ) -> List[float]:

        self._ensure_client()

        assert self._client is not None

        top_n = len(texts)

        if self._top_n is not None:
            top_n = min(top_n, self._top_n)

        response = self._client.rerank(
            model=self._model,
            query=query,
            documents=texts,
            top_n=top_n,
        )

        scores: List[float] = [0.0] * len(texts)

        for item in response.results:
            scores[item.index] = float(item.relevance_score)

        return scores