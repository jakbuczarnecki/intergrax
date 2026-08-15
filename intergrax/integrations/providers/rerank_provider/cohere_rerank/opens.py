# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Only this module may import ``cohere``."""

from __future__ import annotations

import math
from typing import List, Optional

from intergrax.integrations.providers.rerank_provider.cohere_rerank.config import CohereRerankIntegrationConfig
from intergrax.utils import attribute_access

_CLIENT = None


def _client(config: CohereRerankIntegrationConfig):
    global _CLIENT
    import cohere

    if _CLIENT is not None:
        return _CLIENT
    api_key = config.api_key
    if not api_key:
        raise RuntimeError("COHERE_API_KEY not found in environment variables.")
    _CLIENT = cohere.Client(api_key)
    return _CLIENT


def cohere_rerank_scores(
    config: CohereRerankIntegrationConfig,
    query: str,
    texts: List[str],
    *,
    top_n: Optional[int] = None,
) -> List[float]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(texts, list) or any(
        not isinstance(text, str) or not text.strip() for text in texts
    ):
        raise ValueError("texts must contain non-empty strings")
    if top_n is not None and (type(top_n) is not int or top_n <= 0):
        raise ValueError("top_n must be an exact positive int or None")
    if not texts:
        return []
    client = _client(config)
    limit = len(texts)
    effective_top_n = top_n if top_n is not None else config.top_n
    if effective_top_n is not None:
        limit = min(limit, effective_top_n)
    response = client.rerank(
        model=config.model,
        query=query,
        documents=texts,
        top_n=limit,
    )
    response_results = attribute_access.optional(response, "results", None)
    if not isinstance(response_results, list) or not response_results:
        raise ValueError("Cohere response is missing results")
    scores: List[float] = [0.0] * len(texts)
    seen: set[int] = set()
    for item in response_results:
        index = attribute_access.optional(item, "index", None)
        if type(index) is not int or not 0 <= index < len(texts):
            raise ValueError("Cohere response contains an invalid index")
        if index in seen:
            raise ValueError("Cohere response contains a duplicate index")
        seen.add(index)
        raw_score = attribute_access.optional(item, "relevance_score", None)
        if isinstance(raw_score, bool):
            raise TypeError("Cohere response score must be numeric")
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError("Cohere response score must be finite")
        scores[index] = score
    expected = len(texts) if effective_top_n is None else min(effective_top_n, len(texts))
    if len(seen) != expected:
        raise ValueError("Cohere response has an invalid result count")
    return scores
