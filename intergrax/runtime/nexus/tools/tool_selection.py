# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tool subset selection before planner schema export (TOOL-ENG-5)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter

from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.runtime.nexus.config_types import ToolSelectionMode
from intergrax.runtime.nexus.tools.catalog_dispatch import catalog_tool_ids
from intergrax.runtime.nexus.tools.hierarchical_tool_selector import (
    rank_categories,
    rank_categories_with_llm,
    select_tools_hierarchical,
)
from intergrax.runtime.nexus.tools.tool_catalog_embedder import ToolCatalogEmbedder
from intergrax.skills.execution_binding import (
    SkillExecutionPinningStore,
    resolve_bound_skill_pack,
)
from intergrax.skills.registry import SkillProfile, build_registry_from_profile
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.search.keyword_ranking import (
    ToolKeywordSearchDocument,
    score_tool_keyword_document,
    tokenize_tool_search_query,
)


@dataclass(frozen=True, slots=True)
class ToolSelectionContext:
    """Inputs for planner tool subset resolution."""

    registry: ToolRegistry
    query: str
    skill_profile: SkillProfile | None = None
    skill_registry: SkillRegistry | None = None
    skill_pinning_store: SkillExecutionPinningStore | None = None
    tenant_id: str | None = None
    plan_allowed_tool_ids: Sequence[str] | None = None
    top_k: int = 20
    max_hierarchy_passes: int = 2
    embedding_manager: BaseEmbeddingManager | None = None
    hierarchical_llm_category_pass: bool = False
    llm_adapter: LLMAdapter | None = None


@runtime_checkable
class ToolSelectionStrategy(Protocol):
    """Selects catalog ``tool_id`` values exposed to the tool planner."""

    def select_tool_ids(self, ctx: ToolSelectionContext) -> Sequence[str] | None:
        """
        Return planner allow-list.

        ``None`` means no strategy filter (full registry at invoke time).
        """


class StaticAllowListSelectionStrategy:
    """Use explicit plan ``tool_ids`` when present; otherwise no strategy filter."""

    def select_tool_ids(self, ctx: ToolSelectionContext) -> Sequence[str] | None:
        if ctx.plan_allowed_tool_ids is not None:
            return tuple(ctx.plan_allowed_tool_ids)
        return None


class FullCatalogSelectionStrategy:
    """Never narrow — planner sees the full runtime registry."""

    def select_tool_ids(self, ctx: ToolSelectionContext) -> Sequence[str] | None:
        return None


class SkillPackSelectionStrategy:
    """Resolve enabled skills from ``SkillProfile`` to catalog ``tool_ids``."""

    def select_tool_ids(self, ctx: ToolSelectionContext) -> Sequence[str] | None:
        if ctx.skill_profile is None:
            return ()
        skill_registry = ctx.skill_registry or build_registry_from_profile(ctx.skill_profile)
        pack = resolve_bound_skill_pack(
            tenant_id=ctx.tenant_id or "",
            skill_profile=ctx.skill_profile,
            skill_registry=skill_registry,
            pinning_store=ctx.skill_pinning_store,
        )
        if not pack.resolved_skills:
            return ()
        present = tuple(
            sorted(tool_id for tool_id in pack.tool_ids if ctx.registry.has(tool_id))
        )
        return present


class RetrievalTopKSelectionStrategy:
    """Keyword overlap rank over registry metadata; return top-k ``tool_id`` values."""

    def select_tool_ids(self, ctx: ToolSelectionContext) -> Sequence[str] | None:
        registered = list(ctx.registry.list())
        if not registered:
            return ()

        query_tokens = tokenize_tool_search_query(ctx.query)
        if not query_tokens:
            ranked = sorted(registered, key=lambda item: item.contract.tool_id)
            return tuple(rt.contract.tool_id for rt in ranked[: ctx.top_k])

        scored = sorted(
            registered,
            key=lambda item: (-_score_contract(item.contract, query_tokens), item.contract.tool_id),
        )
        return tuple(rt.contract.tool_id for rt in scored[: ctx.top_k])


class HierarchicalToolSelectionStrategy:
    """Category-tree two-pass narrowing (TOOL-ENG-14)."""

    def __init__(self, *, max_category_passes: int = 2) -> None:
        self._max_category_passes = max_category_passes
        self.last_categories: tuple[str, ...] = ()
        self.last_tool_ids: tuple[str, ...] = ()

    @property
    def strategy_id(self) -> str:
        return "hierarchical"

    def select_tool_ids(self, ctx: ToolSelectionContext) -> Sequence[str] | None:
        ranks = rank_categories(
            ctx.registry,
            ctx.query,
            allowed_tool_ids=ctx.plan_allowed_tool_ids,
        )
        self.last_categories = tuple(rank.category for rank in ranks[: ctx.max_hierarchy_passes])
        selected = select_tools_hierarchical(
            ctx.registry,
            ctx.query,
            top_k=ctx.top_k,
            max_category_passes=ctx.max_hierarchy_passes,
            allowed_tool_ids=ctx.plan_allowed_tool_ids,
        )
        self.last_tool_ids = selected
        return selected

    async def select_tool_ids_async(self, ctx: ToolSelectionContext) -> Sequence[str] | None:
        """Optional LLM category pass when ``ctx.hierarchical_llm_category_pass`` is set."""
        if ctx.hierarchical_llm_category_pass and ctx.llm_adapter is not None:
            ranks = await rank_categories_with_llm(
                ctx.registry,
                ctx.query,
                ctx.llm_adapter,
                allowed_tool_ids=ctx.plan_allowed_tool_ids,
            )
        else:
            ranks = rank_categories(
                ctx.registry,
                ctx.query,
                allowed_tool_ids=ctx.plan_allowed_tool_ids,
            )
        self.last_categories = tuple(rank.category for rank in ranks[: ctx.max_hierarchy_passes])
        selected = select_tools_hierarchical(
            ctx.registry,
            ctx.query,
            top_k=ctx.top_k,
            max_category_passes=ctx.max_hierarchy_passes,
            allowed_tool_ids=ctx.plan_allowed_tool_ids,
            category_ranks=ranks,
        )
        self.last_tool_ids = selected
        return selected


class SemanticToolIndexSelectionStrategy:
    """Vector similarity top-k over tool metadata (TOOL-ENG-13)."""

    def __init__(self, embedding_manager: BaseEmbeddingManager | None) -> None:
        self._embedding_manager = embedding_manager
        self.last_ranks: tuple[tuple[str, float], ...] = ()

    @property
    def strategy_id(self) -> str:
        return "semantic"

    def select_tool_ids(self, ctx: ToolSelectionContext) -> Sequence[str] | None:
        if self._embedding_manager is None:
            return ()
        embedder = ToolCatalogEmbedder(self._embedding_manager)
        ranks = embedder.search_registry(
            ctx.registry,
            ctx.query,
            top_k=ctx.top_k,
            allowed_tool_ids=ctx.plan_allowed_tool_ids,
        )
        self.last_ranks = tuple((rank.tool_id, rank.score) for rank in ranks)
        return tuple(rank.tool_id for rank in ranks)


def strategy_for_mode(mode: ToolSelectionMode) -> ToolSelectionStrategy:
    if mode == ToolSelectionMode.SKILL_PACK:
        return SkillPackSelectionStrategy()
    if mode in (ToolSelectionMode.RETRIEVAL_TOP_K, ToolSelectionMode.KEYWORD_TOP_K):
        return RetrievalTopKSelectionStrategy()
    if mode == ToolSelectionMode.FULL_CATALOG:
        return FullCatalogSelectionStrategy()
    if mode == ToolSelectionMode.SEMANTIC:
        return SemanticToolIndexSelectionStrategy(None)
    if mode == ToolSelectionMode.HIERARCHICAL:
        return HierarchicalToolSelectionStrategy()
    return StaticAllowListSelectionStrategy()


def resolve_selection_strategy(
    mode: ToolSelectionMode,
    ctx: ToolSelectionContext,
    *,
    strategy_override: ToolSelectionStrategy | None = None,
    entry_point_strategy_id: str | None = None,
) -> ToolSelectionStrategy:
    if strategy_override is not None:
        return strategy_override
    if entry_point_strategy_id:
        from intergrax.runtime.nexus.tools.tool_selection_registry import (
            load_tool_selection_strategy,
        )

        loaded = load_tool_selection_strategy(entry_point_strategy_id)
        if loaded is not None:
            return loaded
    if mode == ToolSelectionMode.SEMANTIC:
        return SemanticToolIndexSelectionStrategy(ctx.embedding_manager)
    if mode == ToolSelectionMode.HIERARCHICAL:
        return HierarchicalToolSelectionStrategy(max_category_passes=ctx.max_hierarchy_passes)
    return strategy_for_mode(mode)


def strategy_trace_id(strategy: ToolSelectionStrategy) -> str:
    descriptor = type(strategy).__dict__.get("strategy_id")
    if isinstance(descriptor, property):
        value = descriptor.fget(strategy)  # type: ignore[misc]
        if isinstance(value, str):
            return value
    return type(strategy).__name__


def resolve_planner_allowed_tool_ids(
    mode: ToolSelectionMode,
    ctx: ToolSelectionContext,
    *,
    strategy_override: ToolSelectionStrategy | None = None,
    entry_point_strategy_id: str | None = None,
) -> Sequence[str] | None:
    """
    Combine selection strategy with optional plan constraints.

    When both strategy and plan provide ids, returns their intersection.
    Empty intersection → empty tuple (planner schema has no tools).
    """
    strategy = resolve_selection_strategy(
        mode,
        ctx,
        strategy_override=strategy_override,
        entry_point_strategy_id=entry_point_strategy_id,
    )
    strategy_ids = strategy.select_tool_ids(ctx)
    if ctx.plan_allowed_tool_ids is not None:
        plan_ids = tuple(catalog_tool_ids(ctx.plan_allowed_tool_ids))
        if strategy_ids is not None:
            allowed = frozenset(strategy_ids)
            return tuple(tool_id for tool_id in plan_ids if tool_id in allowed)
        return plan_ids

    return strategy_ids


async def resolve_planner_allowed_tool_ids_async(
    mode: ToolSelectionMode,
    ctx: ToolSelectionContext,
    *,
    strategy_override: ToolSelectionStrategy | None = None,
    entry_point_strategy_id: str | None = None,
) -> Sequence[str] | None:
    """Async resolver — enables optional hierarchical LLM category pass (TOOL-MAINT-01b)."""
    strategy = resolve_selection_strategy(
        mode,
        ctx,
        strategy_override=strategy_override,
        entry_point_strategy_id=entry_point_strategy_id,
    )
    if isinstance(strategy, HierarchicalToolSelectionStrategy) and ctx.hierarchical_llm_category_pass:
        strategy_ids = await strategy.select_tool_ids_async(ctx)
    else:
        strategy_ids = strategy.select_tool_ids(ctx)
    if ctx.plan_allowed_tool_ids is not None:
        plan_ids = tuple(catalog_tool_ids(ctx.plan_allowed_tool_ids))
        if strategy_ids is not None:
            allowed = frozenset(strategy_ids)
            return tuple(tool_id for tool_id in plan_ids if tool_id in allowed)
        return plan_ids

    return strategy_ids


def _tool_keyword_search_document(contract: ToolContract) -> ToolKeywordSearchDocument:
    return ToolKeywordSearchDocument(
        tool_id=contract.tool_id,
        text_parts=tuple(
            part
            for part in (
                contract.description,
                contract.description_short,
                " ".join(contract.tags),
                contract.category,
            )
            if part
        ),
    )


def _score_contract(contract: ToolContract, query_tokens: Sequence[str]) -> int:
    return score_tool_keyword_document(
        _tool_keyword_search_document(contract),
        query_tokens,
    )
