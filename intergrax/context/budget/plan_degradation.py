# © Artur Czarnecki. All rights reserved.

"""Plan-native context degradation (CE-02-R1C)."""

from __future__ import annotations

import re
from collections.abc import Sequence

from intergrax.context.budget.degradation import ContextDegradationPolicy, DefaultContextDegradationPolicy
from intergrax.context.contracts import ContextFragmentSource
from intergrax.context.planning import ContextSourceGroup
from intergrax.runtime.nexus.context.context_compiler_models import DegradationStepKind

_CE_CONTEXT_TAG = re.compile(
    r"^\[context:(?P<source>[a-z_]+):[^\]]+\]\s",
    re.IGNORECASE,
)

_OPTIONAL_INJECTION_SOURCES = frozenset(
    {
        ContextFragmentSource.WEBSEARCH,
        ContextFragmentSource.ATTACHMENT,
        ContextFragmentSource.TOOL_OUTPUT,
        ContextFragmentSource.LONGTERM_MEMORY,
        ContextFragmentSource.RAG,
    }
)


def detect_optional_injection_source(content: str) -> ContextFragmentSource | None:
    """Classify non-primary system injections for plan grouping."""
    match = _CE_CONTEXT_TAG.match(content or "")
    if match:
        try:
            return ContextFragmentSource(match.group("source"))
        except ValueError:
            pass
    lowered = (content or "").lower()
    if "long-term memory" in lowered or "user memory" in lowered or "ltm:" in lowered:
        return ContextFragmentSource.LONGTERM_MEMORY
    if "rag context" in lowered or "retrieved documents" in lowered:
        return ContextFragmentSource.RAG
    if "web search" in lowered or "websearch" in lowered:
        return ContextFragmentSource.WEBSEARCH
    if "attachments" in lowered or "session attachments" in lowered:
        return ContextFragmentSource.ATTACHMENT
    if "tool" in lowered and "context" in lowered:
        return ContextFragmentSource.TOOL_OUTPUT
    return None


def _total_selected_tokens(
    selected_ids: set[str],
    groups_by_id: dict[str, ContextSourceGroup],
) -> int:
    return sum(groups_by_id[group_id].token_estimate for group_id in selected_ids)


def _group_eligible_optional_injection(
    group: ContextSourceGroup,
    *,
    prefer_longterm_memory: bool,
    prefer_rag_when_enabled: bool,
) -> bool:
    if not group.droppable or group.required or group.protected:
        return False
    source = group.source
    if source is ContextFragmentSource.WEBSEARCH:
        return True
    if source in {
        ContextFragmentSource.ATTACHMENT,
        ContextFragmentSource.TOOL_OUTPUT,
    }:
        return True
    if source is ContextFragmentSource.LONGTERM_MEMORY:
        return not prefer_longterm_memory
    if source is ContextFragmentSource.RAG:
        return not prefer_rag_when_enabled
    return source in _OPTIONAL_INJECTION_SOURCES


def _apply_drop_optional_injections(
    *,
    selected_ids: set[str],
    excluded_ids: set[str],
    groups_by_id: dict[str, ContextSourceGroup],
    prefer_longterm_memory: bool,
    prefer_rag_when_enabled: bool,
) -> bool:
    dropped = False
    for group_id in tuple(selected_ids):
        group = groups_by_id[group_id]
        if not _group_eligible_optional_injection(
            group,
            prefer_longterm_memory=prefer_longterm_memory,
            prefer_rag_when_enabled=prefer_rag_when_enabled,
        ):
            continue
        selected_ids.remove(group_id)
        excluded_ids.add(group_id)
        dropped = True
    return dropped


def _apply_drop_lowest_scored(
    *,
    selected_ids: set[str],
    excluded_ids: set[str],
    groups_by_id: dict[str, ContextSourceGroup],
    group_scores: dict[str, float],
) -> bool:
    droppable = [
        group_id
        for group_id in selected_ids
        if groups_by_id[group_id].droppable
        and not groups_by_id[group_id].required
        and not groups_by_id[group_id].protected
    ]
    if not droppable:
        return False
    drop_id = min(droppable, key=lambda group_id: group_scores.get(group_id, 0.5))
    selected_ids.remove(drop_id)
    excluded_ids.add(drop_id)
    return True


def apply_plan_degradation(
    *,
    ordered_group_ids: Sequence[str],
    groups_by_id: dict[str, ContextSourceGroup],
    group_scores: dict[str, float],
    budget_tokens: int,
    prefer_longterm_memory: bool,
    prefer_rag_when_enabled: bool,
    policy: ContextDegradationPolicy | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Reduce selected groups per degradation policy until within budget."""
    active_policy = policy or DefaultContextDegradationPolicy()
    selected_set = set(ordered_group_ids)
    excluded_set: set[str] = set()
    applied: list[str] = []

    if _total_selected_tokens(selected_set, groups_by_id) <= budget_tokens:
        return (
            tuple(ordered_group_ids),
            (),
            (),
        )

    while _total_selected_tokens(selected_set, groups_by_id) > budget_tokens:
        progressed = False
        for step in active_policy.ladder_order():
            if step == DegradationStepKind.FULL:
                continue
            effective_step = step
            if step == DegradationStepKind.REDUCE_INJECTION_BLOCKS:
                effective_step = DegradationStepKind.DROP_LOWEST_SCORED
            if effective_step not in {
                DegradationStepKind.DROP_OPTIONAL_INJECTIONS,
                DegradationStepKind.DROP_LOWEST_SCORED,
            }:
                continue

            if _total_selected_tokens(selected_set, groups_by_id) <= budget_tokens:
                break

            changed = False
            if effective_step == DegradationStepKind.DROP_OPTIONAL_INJECTIONS:
                changed = _apply_drop_optional_injections(
                    selected_ids=selected_set,
                    excluded_ids=excluded_set,
                    groups_by_id=groups_by_id,
                    prefer_longterm_memory=prefer_longterm_memory,
                    prefer_rag_when_enabled=prefer_rag_when_enabled,
                )
            else:
                changed = _apply_drop_lowest_scored(
                    selected_ids=selected_set,
                    excluded_ids=excluded_set,
                    groups_by_id=groups_by_id,
                    group_scores=group_scores,
                )

            if changed:
                applied.append(effective_step.value)
                progressed = True
            if _total_selected_tokens(selected_set, groups_by_id) <= budget_tokens:
                break
        if not progressed:
            break

    selected_ids = tuple(group_id for group_id in ordered_group_ids if group_id in selected_set)
    excluded_ids = tuple(group_id for group_id in ordered_group_ids if group_id in excluded_set)
    return selected_ids, excluded_ids, tuple(applied)
