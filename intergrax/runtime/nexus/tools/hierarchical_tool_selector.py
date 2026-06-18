# © Artur Czarnecki. All rights reserved.

"""Category-tree hierarchical tool selection (TOOL-ENG-14)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import re

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class CategoryRank:
    category: str
    score: int
    tool_ids: tuple[str, ...]


def _category_key(contract: ToolContract) -> str:
    return contract.category.strip() or "_uncategorized"


def _query_tokens(query: str) -> tuple[str, ...]:
    return tuple(token for token in query.lower().split() if len(token) > 2)


def _score_text(haystack: str, query_tokens: Sequence[str]) -> int:
    lowered = haystack.lower()
    return sum(1 for token in query_tokens if token in lowered)


def _contract_haystack(contract: ToolContract) -> str:
    return " ".join(
        part
        for part in (
            contract.tool_id,
            contract.description,
            contract.description_short or "",
            " ".join(contract.tags),
            contract.category,
        )
        if part
    )


def rank_categories(
    registry: ToolRegistry,
    query: str,
    *,
    allowed_tool_ids: Sequence[str] | None = None,
) -> tuple[CategoryRank, ...]:
    """Pass 1 — rank category buckets by query overlap."""
    allowed = frozenset(allowed_tool_ids) if allowed_tool_ids else None
    buckets: dict[str, list[str]] = {}
    for registered in registry.list():
        tool_id = registered.contract.tool_id
        if allowed is not None and tool_id not in allowed:
            continue
        key = _category_key(registered.contract)
        buckets.setdefault(key, []).append(tool_id)

    query_tokens = _query_tokens(query)
    ranked: list[CategoryRank] = []
    for category, tool_ids in buckets.items():
        haystack = category
        for tool_id in tool_ids:
            haystack += " " + _contract_haystack(registry.get(tool_id).contract)
        score = _score_text(haystack, query_tokens) if query_tokens else 0
        ranked.append(CategoryRank(category=category, score=score, tool_ids=tuple(sorted(tool_ids))))

    return tuple(sorted(ranked, key=lambda item: (-item.score, item.category)))


def select_tools_hierarchical(
    registry: ToolRegistry,
    query: str,
    *,
    top_k: int,
    max_category_passes: int = 2,
    allowed_tool_ids: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """
    Two-pass hierarchical narrowing: category rank → tool rank within branches.

    ``max_category_passes`` bounds how many top-scoring categories contribute tools.
    """
    if top_k < 1:
        return ()

    category_ranks = rank_categories(registry, query, allowed_tool_ids=allowed_tool_ids)
    if not category_ranks:
        return ()

    pass_budget = max(1, max_category_passes)
    selected_categories = category_ranks[:pass_budget]
    candidate_ids: list[str] = []
    for rank in selected_categories:
        candidate_ids.extend(rank.tool_ids)

    if not candidate_ids:
        return ()

    query_tokens = _query_tokens(query)
    if not query_tokens:
        return tuple(sorted(candidate_ids)[:top_k])

    scored = sorted(
        candidate_ids,
        key=lambda tool_id: (
            -_score_text(_contract_haystack(registry.get(tool_id).contract), query_tokens),
            tool_id,
        ),
    )
    return tuple(scored[:top_k])


async def rank_categories_with_llm(
    registry: ToolRegistry,
    query: str,
    llm: LLMAdapter,
    *,
    allowed_tool_ids: Sequence[str] | None = None,
) -> tuple[CategoryRank, ...]:
    """
    Optional LLM category pass (TOOL-MAINT-01 / ADR-TOOL-005 v2).

    Falls back to deterministic ``rank_categories`` when the model output is unusable.
    """
    deterministic = rank_categories(registry, query, allowed_tool_ids=allowed_tool_ids)
    categories = tuple(rank.category for rank in deterministic[:8])
    if not categories:
        return deterministic

    prompt = (
        "Rank integration tool categories by relevance to the user query. "
        f"Query: {query!r}\n"
        f"Categories: {list(categories)}\n"
        'Return JSON: {"ordered_categories": ["category", ...]}'
    )
    try:
        raw = await llm.generate(prompt)
    except Exception:
        return deterministic
    return _order_ranks_from_llm(raw, deterministic)


def _order_ranks_from_llm(
    raw: str,
    ranks: Sequence[CategoryRank],
) -> tuple[CategoryRank, ...]:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match is None:
        return tuple(ranks)
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return tuple(ranks)
    ordered = payload.get("ordered_categories")
    if not isinstance(ordered, list):
        return tuple(ranks)
    by_category = {rank.category: rank for rank in ranks}
    reordered: list[CategoryRank] = []
    for item in ordered:
        key = str(item).strip()
        if key in by_category:
            reordered.append(by_category.pop(key))
    reordered.extend(by_category.values())
    return tuple(reordered)
