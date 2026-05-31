# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Only this module may import ``cohere``."""

from __future__ import annotations

from typing import List, Optional

from intergrax.integrations.providers.rerank_provider.cohere_rerank.config import CohereRerankIntegrationConfig

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
    scores: List[float] = [0.0] * len(texts)
    for item in response.results:
        scores[item.index] = float(item.relevance_score)
    return scores
