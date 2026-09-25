# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import os
from typing import Optional

from intergrax.integrations.contracts.rerank_provider import RerankProvider
from intergrax.rag.rerankers.integration.resolver import resolve_rerank_provider
from intergrax.rag.rerankers.providers._api_reranker_base import _APIRerankerBase


class CohereReranker(_APIRerankerBase):

    DEFAULT_MODEL = "rerank-english-v3.0"
    ENV_MODEL = "INTERGRAX_DEFAULT_COHERE_RERANK_MODEL"

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> None:
        env_model = os.getenv(self.ENV_MODEL)
        self._model = model or env_model or self.DEFAULT_MODEL
        self._top_n = top_n

    @classmethod
    def name(cls) -> str:
        return "cohere"

    def _resolve_provider(self) -> RerankProvider:
        return resolve_rerank_provider(
            "cohere_rerank",
            model=self._model,
            top_n=self._top_n,
        )
