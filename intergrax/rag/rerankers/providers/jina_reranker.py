# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import os
from typing import List, Optional

import requests

from intergrax.rag.rerankers.providers._api_reranker_base import _APIRerankerBase


class JinaReranker(_APIRerankerBase):

    DEFAULT_MODEL = "jina-reranker-v1-base-en"

    ENV_MODEL = "INTERGRAX_DEFAULT_JINA_RERANK_MODEL"
    ENV_API_KEY = "JINA_API_KEY"

    API_URL = "https://api.jina.ai/v1/rerank"

    def __init__(
        self,
        *,
        model: Optional[str] = None,
    ) -> None:

        env_model = os.getenv(self.ENV_MODEL)

        resolved_model = model or env_model or self.DEFAULT_MODEL

        # lazy initialization
        self._api_key: Optional[str] = None
        self._model = resolved_model

    @classmethod
    def name(cls) -> str:
        return "jina"

    def _ensure_api_key(self) -> None:

        if self._api_key is not None:
            return

        env_api_key = os.getenv(self.ENV_API_KEY)

        if not env_api_key:
            raise RuntimeError(
                "JINA_API_KEY not found in environment variables."
            )

        self._api_key = env_api_key

    def _score(
        self,
        query: str,
        texts: List[str],
    ) -> List[float]:

        self._ensure_api_key()

        assert self._api_key is not None

        payload = {
            "model": self._model,
            "query": query,
            "documents": texts,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            self.API_URL,
            json=payload,
            headers=headers,
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        scores: List[float] = [0.0] * len(texts)

        for item in data["results"]:
            scores[item["index"]] = float(item["relevance_score"])

        return scores