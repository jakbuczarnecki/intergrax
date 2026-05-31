# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Pluggable query expansion for multi-query retrieval."""

from __future__ import annotations

from typing import List, Optional, Protocol

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter


class QueryExpander(Protocol):
    def expand(self, query: str, *, num_queries: int) -> List[str]: ...


class DeterministicQueryExpander:
    """Default expansion without LLM cost."""

    def expand(self, query: str, *, num_queries: int) -> List[str]:
        variants = {query.strip()}
        words = query.split()
        if len(words) > 2:
            variants.add(" ".join(words[:2]))
            variants.add(" ".join(words[-2:]))
        return list(variants)[: max(1, num_queries)]


class LlmQueryExpander:
    """LLM paraphrases when an adapter is injected (no provider hardcoding)."""

    def __init__(self, llm: LLMAdapter) -> None:
        self._llm = llm

    def expand(self, query: str, *, num_queries: int) -> List[str]:
        prompt = (
            f"Generate {max(1, num_queries - 1)} short search query variants for:\n{query}\n"
            "One variant per line, no numbering."
        )
        try:
            response = self._llm.generate_messages(
                [ChatMessage(role="user", content=prompt)],
                run_id="rag-query-expand",
            )
            lines = [ln.strip() for ln in (response.content or "").splitlines() if ln.strip()]
            out = [query]
            for line in lines:
                if line not in out:
                    out.append(line)
            return out[:num_queries]
        except Exception:
            return DeterministicQueryExpander().expand(query, num_queries=num_queries)


def query_expander_from_profile(
    *,
    mode: str,
    llm: Optional[LLMAdapter] = None,
) -> QueryExpander:
    if mode == "llm" and llm is not None:
        return LlmQueryExpander(llm)
    return DeterministicQueryExpander()
