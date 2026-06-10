# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Lightweight query complexity routing (adaptive RAG tier selection)."""

from __future__ import annotations

from typing import Literal, Optional

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.routing.llm_tier_classifier import classify_route_tier_with_llm

RouteTier = Literal["fast", "standard", "deep"]
RouteClassifier = Literal["heuristic", "llm"]


class QueryRouter:
    """
    Classify queries into retrieval tiers when ``route_mode=auto``.

    - *fast*: short, factual lookups → cheaper retriever
    - *standard*: default hybrid + rerank
    - *deep*: long or multi-clause queries → fusion / multi-strategy retriever

    When ``llm_route_enabled`` and an ``LLMAdapter`` is injected, an LLM classifier
  runs before the word-count heuristic (with heuristic fallback on failure).
    """

    def __init__(
        self,
        profile: RagProfile,
        *,
        llm: Optional[LLMAdapter] = None,
    ) -> None:
        self._profile = profile
        self._llm = llm
        self._last_route_classifier: RouteClassifier = "heuristic"

    @property
    def last_route_classifier(self) -> RouteClassifier:
        return self._last_route_classifier

    def route(self, query_text: str) -> RouteTier:
        if self._profile.route_mode == "off":
            self._last_route_classifier = "heuristic"
            return "standard"

        if self._profile.llm_route_enabled and self._llm is not None:
            llm_tier = classify_route_tier_with_llm(self._llm, query_text)
            if llm_tier is not None:
                self._last_route_classifier = "llm"
                return llm_tier

        self._last_route_classifier = "heuristic"
        return self._heuristic_route(query_text)

    def _heuristic_route(self, query_text: str) -> RouteTier:
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
