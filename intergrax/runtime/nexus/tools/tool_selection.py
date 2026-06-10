# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tool subset selection before planner schema export (TOOL-ENG-5)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.runtime.nexus.config_types import ToolSelectionMode
from intergrax.runtime.nexus.tools.catalog_dispatch import catalog_tool_ids
from intergrax.skills.registry import SkillProfile, build_registry_from_profile, enabled_skill_ids_for_profile
from intergrax.skills.resolver import SkillResolver
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class ToolSelectionContext:
    """Inputs for planner tool subset resolution."""

    registry: ToolRegistry
    query: str
    skill_profile: SkillProfile | None = None
    plan_allowed_tool_ids: Sequence[str] | None = None
    top_k: int = 20


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
        if ctx.plan_allowed_tool_ids:
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
        skill_ids = enabled_skill_ids_for_profile(ctx.skill_profile)
        if not skill_ids:
            return ()
        skill_registry = build_registry_from_profile(ctx.skill_profile)
        pack = SkillResolver(skill_registry, tool_registry=None).resolve(skill_ids)
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

        query_tokens = _query_tokens(ctx.query)
        if not query_tokens:
            ranked = sorted(registered, key=lambda item: item.contract.tool_id)
            return tuple(rt.contract.tool_id for rt in ranked[: ctx.top_k])

        scored = sorted(
            registered,
            key=lambda item: (-_score_contract(item.contract, query_tokens), item.contract.tool_id),
        )
        return tuple(rt.contract.tool_id for rt in scored[: ctx.top_k])


def strategy_for_mode(mode: ToolSelectionMode) -> ToolSelectionStrategy:
    if mode == ToolSelectionMode.SKILL_PACK:
        return SkillPackSelectionStrategy()
    if mode == ToolSelectionMode.RETRIEVAL_TOP_K:
        return RetrievalTopKSelectionStrategy()
    if mode == ToolSelectionMode.FULL_CATALOG:
        return FullCatalogSelectionStrategy()
    return StaticAllowListSelectionStrategy()


def resolve_planner_allowed_tool_ids(
    mode: ToolSelectionMode,
    ctx: ToolSelectionContext,
) -> Sequence[str] | None:
    """
    Combine selection strategy with optional plan constraints.

    When both strategy and plan provide ids, returns their intersection.
    Empty intersection → empty tuple (planner schema has no tools).
    """
    strategy_ids = strategy_for_mode(mode).select_tool_ids(ctx)
    plan_ids = tuple(catalog_tool_ids(ctx.plan_allowed_tool_ids or ()))

    if plan_ids and strategy_ids is not None:
        allowed = frozenset(strategy_ids)
        intersected = tuple(tool_id for tool_id in plan_ids if tool_id in allowed)
        return intersected

    if plan_ids:
        return plan_ids
    return strategy_ids


def _query_tokens(query: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in query.lower().split()
        if len(token) > 2
    )


def _score_contract(contract: ToolContract, query_tokens: Sequence[str]) -> int:
    haystack = " ".join(
        part
        for part in (
            contract.tool_id,
            contract.description,
            contract.description_short or "",
            " ".join(contract.tags),
            contract.category,
        )
        if part
    ).lower()
    return sum(1 for token in query_tokens if token in haystack)
