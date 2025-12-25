# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from intergrax.llm_adapters.llm_adapter import LLMAdapter


class WebSearchStrategyType(str, Enum):
    SERP_ONLY = "SERP_ONLY"
    URL_CONTEXT_TOPK = "URL_CONTEXT_TOPK"
    CHUNK_RERANK = "CHUNK_RERANK"
    MAP_REDUCE = "MAP_REDUCE"


@dataclass
class WebSearchLLMConfig:
    """
    LLMs used by websearch grounding steps.
    You can point all of them to the same adapter if needed,
    but it stays explicit and configurable.
    """
    # Used to create per-URL fact cards / mini-summaries
    map_adapter: Optional[LLMAdapter] = None

    # Used to synthesize final grounded context from fact cards
    reduce_adapter: Optional[LLMAdapter] = None

    # Optional: for LLM-based reranking of chunks/snippets
    rerank_adapter: Optional[LLMAdapter] = None


@dataclass
class WebSearchConfig:
    strategy: WebSearchStrategyType = WebSearchStrategyType.URL_CONTEXT_TOPK

    max_docs: int = 8

    # budgets (heuristic token budgeting at this layer is ok; strict budgeting can be enforced by adapter)
    token_budget_total: int = 1800
    token_budget_per_doc: int = 450

    # chunking/rerank knobs
    chunk_chars: int = 1500
    max_chunks_total: int = 10

    run_id: Optional[str] = None

    llm: WebSearchLLMConfig = field(default_factory=WebSearchLLMConfig)
