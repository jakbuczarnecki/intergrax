# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import os
from typing import List, Optional

from intergrax.integrations.registry.slugs import IntegrationSlug
from intergrax.rag.rerankers.integration.resolver import rerank_scores
from intergrax.rag.rerankers.providers._api_reranker_base import _APIRerankerBase


class JinaReranker(_APIRerankerBase):

    DEFAULT_MODEL = "jina-reranker-v1-base-en"
    ENV_MODEL = "INTERGRAX_DEFAULT_JINA_RERANK_MODEL"

    def __init__(
        self,
        *,
        model: Optional[str] = None,
    ) -> None:
        env_model = os.getenv(self.ENV_MODEL)
        self._model = model or env_model or self.DEFAULT_MODEL

    @classmethod
    def name(cls) -> str:
        return "jina"

    def _score(
        self,
        query: str,
        texts: List[str],
    ) -> List[float]:
        return rerank_scores(
            IntegrationSlug.JINA_RERANK,
            query,
            texts,
            model=self._model,
        )
