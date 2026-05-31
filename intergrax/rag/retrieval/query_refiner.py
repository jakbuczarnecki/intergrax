# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Pluggable query refinement for agentic retrieval (deterministic or LLM)."""

from __future__ import annotations

from typing import Literal, Optional, Protocol

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.retrieval_result import RetrievalResult

AgenticQueryMode = Literal["deterministic", "llm"]


class QueryRefiner(Protocol):
    def refine(self, query: str, result: RetrievalResult) -> str: ...


class DeterministicQueryRefiner:
    def refine(self, query: str, result: RetrievalResult) -> str:
        if not result.chunks:
            return query
        terms: list[str] = []
        for chunk in result.chunks[:2]:
            words = [w for w in chunk.text.split() if len(w) > 4][:3]
            terms.extend(words)
        if terms:
            return f"{query} {' '.join(dict.fromkeys(terms))}"
        return query


class LlmQueryRefiner:
    def __init__(self, llm: LLMAdapter) -> None:
        self._llm = llm

    def refine(self, query: str, result: RetrievalResult) -> str:
        context = "\n".join(c.text[:200] for c in (result.chunks or [])[:3])
        prompt = (
            "Rewrite the search query to improve document retrieval. "
            "Output only the new query, no explanation.\n\n"
            f"Original: {query}\n"
            f"Retrieved context (may be incomplete):\n{context or '(none)'}"
        )
        try:
            response = self._llm.generate_messages(
                [ChatMessage(role="user", content=prompt)],
                run_id="rag-agentic-refine",
            )
            refined = (response.content or "").strip()
            return refined or query
        except Exception:
            return DeterministicQueryRefiner().refine(query, result)


def resolve_query_refiner(
    profile: RagProfile,
    *,
    llm: Optional[LLMAdapter] = None,
) -> QueryRefiner:
    mode: AgenticQueryMode = profile.agentic_query_mode  # type: ignore[attr-defined]
    if mode == "llm" and llm is not None:
        return LlmQueryRefiner(llm)
    return DeterministicQueryRefiner()
