# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Optional LLM tier classifier for adaptive RAG routing (M-RAG.32)."""

from __future__ import annotations

from typing import Literal, Optional

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter

RouteTier = Literal["fast", "standard", "deep"]

_VALID_TIERS: frozenset[str] = frozenset({"fast", "standard", "deep"})


def classify_route_tier_with_llm(
    llm: LLMAdapter,
    query_text: str,
    *,
    run_id: str = "rag-route-tier",
) -> Optional[RouteTier]:
    """
    Ask the injected adapter to classify query complexity.

    Returns ``None`` when the model response is unusable (caller should fall back).
    """
    text = (query_text or "").strip()
    if not text:
        return None

    prompt = (
        "Classify the search query complexity for document retrieval routing.\n"
        "Reply with exactly one word: fast, standard, or deep.\n"
        "- fast: short factual lookup\n"
        "- standard: typical question answering\n"
        "- deep: complex, multi-part, ambiguous, or analytical query\n\n"
        f"Query: {text}"
    )
    try:
        response = llm.generate_messages(
            [ChatMessage(role="user", content=prompt)],
            run_id=run_id,
        )
    except Exception:
        return None

    return parse_route_tier_response(response.content or "")


def parse_route_tier_response(content: str) -> Optional[RouteTier]:
    normalized = (content or "").strip().lower()
    if not normalized:
        return None

    first_token = normalized.split()[0].strip(".,;:\"'")
    if first_token in _VALID_TIERS:
        return first_token  # type: ignore[return-value]

    for tier in ("deep", "standard", "fast"):
        if tier in normalized:
            return tier  # type: ignore[return-value]
    return None
