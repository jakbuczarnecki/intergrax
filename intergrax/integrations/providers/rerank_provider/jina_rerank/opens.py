# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Only this module may import HTTP client for Jina rerank API."""

from __future__ import annotations

import math
from typing import List

import requests

from intergrax.integrations.providers.rerank_provider.jina_rerank.config import JinaRerankIntegrationConfig


def jina_rerank_scores(
    config: JinaRerankIntegrationConfig,
    query: str,
    texts: List[str],
) -> List[float]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(texts, list) or any(
        not isinstance(text, str) or not text.strip() for text in texts
    ):
        raise ValueError("texts must contain non-empty strings")
    if not texts:
        return []
    api_key = config.api_key
    if not api_key:
        raise RuntimeError("JINA_API_KEY not found in environment variables.")
    payload = {"model": config.model, "query": query, "documents": texts}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(config.api_url, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()
    response_results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(response_results, list) or not response_results:
        raise ValueError("Jina response is missing results")
    scores: List[float] = [0.0] * len(texts)
    seen: set[int] = set()
    for item in response_results:
        if not isinstance(item, dict):
            raise ValueError("Jina response contains an invalid result")
        index = item.get("index")
        if type(index) is not int or not 0 <= index < len(texts):
            raise ValueError("Jina response contains an invalid index")
        if index in seen:
            raise ValueError("Jina response contains a duplicate index")
        seen.add(index)
        raw_score = item.get("relevance_score")
        if isinstance(raw_score, bool) or raw_score is None:
            raise ValueError("Jina response is missing a score")
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError("Jina response score must be finite")
        scores[index] = score
    if len(seen) != len(texts):
        raise ValueError("Jina response has an invalid result count")
    return scores
