# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Only this module may import HTTP client for Jina rerank API."""

from __future__ import annotations

from typing import List

import requests

from intergrax.integrations.providers.rerank_provider.jina_rerank.config import JinaRerankIntegrationConfig


def jina_rerank_scores(
    config: JinaRerankIntegrationConfig,
    query: str,
    texts: List[str],
) -> List[float]:
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
    scores: List[float] = [0.0] * len(texts)
    for item in data["results"]:
        scores[item["index"]] = float(item["relevance_score"])
    return scores
