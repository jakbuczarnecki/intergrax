# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Lightweight query complexity routing (adaptive RAG tier selection)."""

from __future__ import annotations

from typing import Literal

from intergrax.rag.profiles.rag_profile import RagProfile

RouteTier = Literal["fast", "standard", "deep"]


class QueryRouter:
    """
    Classify queries into retrieval tiers without LLM cost when ``route_mode=auto``.

    - *fast*: short, factual lookups → cheaper retriever
    - *standard*: default hybrid + rerank
    - *deep*: long or multi-clause queries → fusion / multi-strategy retriever
    """

    def __init__(self, profile: RagProfile) -> None:
        self._profile = profile

    def route(self, query_text: str) -> RouteTier:
        if self._profile.route_mode == "off":
            return "standard"

        text = (query_text or "").strip()
        if not text:
            return "standard"

        words = text.split()
        if len(words) <= 4:
            return "fast"

        if len(words) >= self._profile.deep_query_min_words or " and " in text.lower():
            return "deep"

        if text.count("?") >= 2:
            return "deep"

        return "standard"
