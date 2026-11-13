# © Artur Czarnecki. All rights reserved.

"""Ordered context degradation ladder (Phase MEM-DEPTH-1.2)."""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, replace

from intergrax.context.budget.contracts import (
    DEFAULT_CONTEXT_DEGRADATION_LADDER_ORDER,
    DegradationStepKind,
)
from intergrax.context.budget.mandatory_base_messages import complete_tool_call_group_spans
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.context.context_budget import (
    ContextBudgetPolicy,
    trim_message_to_budget_tokenizer_aware,
)
from intergrax.runtime.nexus.context.context_compiler_models import (
    ContextCandidate,
    ContextCandidateSource,
)


@dataclass(frozen=True, slots=True)
class DegradationApplyResult:
    messages: list[ChatMessage]
    step: DegradationStepKind
    bytes_removed: int
    tokens_removed: int


def _sum_tokens(candidates: Sequence[ContextCandidate]) -> int:
    return sum(candidate.token_estimate for candidate in candidates)


def _rebuild_messages(
    messages: list[ChatMessage],
    keep_indices: set[int],
) -> list[ChatMessage]:
    return [message for index, message in enumerate(messages) if index in keep_indices]


def tool_call_group_blocked_indices(
    messages: Sequence[ChatMessage],
    mandatory: Collection[int],
) -> frozenset[int]:
    """Indices of complete tool-call groups that contain a mandatory message (never droppable)."""
    blocked: set[int] = set()
    for start, end in complete_tool_call_group_spans(messages):
        group = range(start, end)
        if any(index in mandatory for index in group):
            blocked.update(group)
    return frozenset(blocked)


def expand_to_tool_call_groups(
    messages: Sequence[ChatMessage],
    drop: Collection[int],
) -> set[int]:
    """Grow a drop set to whole tool-call groups so a call or its result is never orphaned."""
    expanded = set(drop)
    for start, end in complete_tool_call_group_spans(messages):
        if any(index in drop for index in range(start, end)):
            expanded.update(range(start, end))
    return expanded


def _hard_trim_messages(
    messages: list[ChatMessage],
    candidates: Sequence[ContextCandidate],
    *,
    budget_tokens: int,
    count_tokens: Callable[[str], int],
) -> list[ChatMessage]:
    """Truncate content only (identity and tool-call linkage kept); drop unfittable pairs atomically.

    Mandatory messages and the final user turn are never touched. The remaining token slack is
    shared out fairly among the other messages; when even one token per message no longer fits,
    the oldest complete tool-call group is dropped as a unit (PDC-D2-R011).
    """
    fixed = {candidate.message_index for candidate in candidates if candidate.mandatory}
    if messages and messages[-1].role == "user":
        fixed.add(len(messages) - 1)
    sizes = [count_tokens(message.content or "") for message in messages]
    alive = set(range(len(messages)))
    spans = complete_tool_call_group_spans(messages)

    while True:
        flexible = [index for index in sorted(alive) if index not in fixed]
        if not flexible:
            break
        slack = budget_tokens - sum(sizes[index] for index in alive if index in fixed)
        if slack >= len(flexible):
            break
        victim = next(
            (
                set(range(start, end))
                for start, end in spans
                if all(index in alive and index not in fixed for index in range(start, end))
            ),
            None,
        )
        if victim is None:
            break
        alive -= victim

    flexible = [index for index in sorted(alive) if index not in fixed]
    slack = max(0, budget_tokens - sum(sizes[index] for index in alive if index in fixed))
    limits: dict[int, int] = {}
    remaining = slack
    ordered = sorted(flexible, key=lambda index: sizes[index])
    for position, index in enumerate(ordered):
        share = max(1, remaining // (len(ordered) - position))
        limits[index] = min(sizes[index], share)
        remaining = max(0, remaining - limits[index])

    result: list[ChatMessage] = []
    for index, message in enumerate(messages):
        if index not in alive:
            continue
        limit = limits.get(index)
        if limit is None or sizes[index] <= limit:
            result.append(message)
            continue
        policy = ContextBudgetPolicy(max_chars=limit * 4, max_tokens_estimate=limit)
        trimmed = trim_message_to_budget_tokenizer_aware(
            message.content or "",
            policy,
            count_tokens=count_tokens,
        )
        result.append(replace(message, content=trimmed.message))
    return result


def _classify_removable_indices(
    candidates: Sequence[ContextCandidate],
    *,
    prefer_longterm_memory: bool,
    prefer_rag_when_enabled: bool,
) -> set[int]:
    removable: set[int] = set()
    for candidate in candidates:
        if candidate.mandatory:
            continue
        source = candidate.source
        if source == ContextCandidateSource.WEBSEARCH and not prefer_rag_when_enabled:
            removable.add(candidate.message_index)
        elif source in {
            ContextCandidateSource.WEBSEARCH,
            ContextCandidateSource.ATTACHMENTS,
            ContextCandidateSource.TOOLS,
        }:
            removable.add(candidate.message_index)
        elif source == ContextCandidateSource.LONGTERM_MEMORY and not prefer_longterm_memory:
            removable.add(candidate.message_index)
        elif source == ContextCandidateSource.RAG and not prefer_rag_when_enabled:
            removable.add(candidate.message_index)
    return removable


def apply_degradation_step(
    *,
    messages: list[ChatMessage],
    candidates: Sequence[ContextCandidate],
    step: DegradationStepKind,
    budget_tokens: int,
    prefer_longterm_memory: bool,
    prefer_rag_when_enabled: bool,
    count_tokens: Callable[[str], int],
) -> DegradationApplyResult | None:
    """Apply one ladder step; return None when step cannot reduce further."""
    current_tokens = _sum_tokens(candidates)
    if current_tokens <= budget_tokens:
        return DegradationApplyResult(
            messages=list(messages),
            step=DegradationStepKind.FULL,
            bytes_removed=0,
            tokens_removed=0,
        )

    if step == DegradationStepKind.DROP_OPTIONAL_INJECTIONS:
        removable = _classify_removable_indices(
            candidates,
            prefer_longterm_memory=prefer_longterm_memory,
            prefer_rag_when_enabled=prefer_rag_when_enabled,
        )
        blocked = tool_call_group_blocked_indices(
            messages,
            {candidate.message_index for candidate in candidates if candidate.mandatory},
        )
        removable = expand_to_tool_call_groups(messages, removable - blocked) - blocked
        if not removable:
            return None
        keep = {candidate.message_index for candidate in candidates} - removable
        new_messages = _rebuild_messages(messages, keep)
        removed_chars = sum(len(messages[i].content or "") for i in removable)
        new_candidates = [c for c in candidates if c.message_index in keep]
        return DegradationApplyResult(
            messages=new_messages,
            step=step,
            bytes_removed=removed_chars,
            tokens_removed=max(0, current_tokens - _sum_tokens(new_candidates)),
        )

    if step == DegradationStepKind.TRUNCATE_OLDEST_HISTORY:
        blocked = tool_call_group_blocked_indices(
            messages,
            {candidate.message_index for candidate in candidates if candidate.mandatory},
        )
        history_indices = [
            candidate.message_index
            for candidate in candidates
            if candidate.source == ContextCandidateSource.SESSION_HISTORY
            and not candidate.mandatory
            and candidate.message_index not in blocked
        ]
        if not history_indices:
            return None
        drop_indices = expand_to_tool_call_groups(messages, {min(history_indices)})
        keep = {candidate.message_index for candidate in candidates if candidate.message_index not in drop_indices}
        new_messages = _rebuild_messages(messages, keep)
        removed_chars = sum(len(messages[i].content or "") for i in drop_indices)
        new_candidates = [c for c in candidates if c.message_index in keep]
        return DegradationApplyResult(
            messages=new_messages,
            step=step,
            bytes_removed=removed_chars,
            tokens_removed=max(0, current_tokens - _sum_tokens(new_candidates)),
        )

    if step == DegradationStepKind.DROP_LOWEST_SCORED:
        blocked = tool_call_group_blocked_indices(
            messages,
            {candidate.message_index for candidate in candidates if candidate.mandatory},
        )
        droppable = [
            candidate for candidate in candidates if not candidate.mandatory and candidate.message_index not in blocked
        ]
        if not droppable:
            return None
        drop = min(droppable, key=lambda c: c.score)
        drop_indices = expand_to_tool_call_groups(messages, {drop.message_index})
        keep = {candidate.message_index for candidate in candidates if candidate.message_index not in drop_indices}
        new_messages = _rebuild_messages(messages, keep)
        removed_chars = sum(len(messages[i].content or "") for i in drop_indices)
        new_candidates = [c for c in candidates if c.message_index in keep]
        return DegradationApplyResult(
            messages=new_messages,
            step=step,
            bytes_removed=removed_chars,
            tokens_removed=max(0, current_tokens - _sum_tokens(new_candidates)),
        )

    if step == DegradationStepKind.TOKENIZER_HARD_TRIM:
        total_chars_before = sum(len(m.content or "") for m in messages)
        trimmed_messages = _hard_trim_messages(
            messages,
            candidates,
            budget_tokens=budget_tokens,
            count_tokens=count_tokens,
        )
        total_chars_after = sum(len(m.content or "") for m in trimmed_messages)
        tokens_after = sum(count_tokens(m.content or "") for m in trimmed_messages)
        return DegradationApplyResult(
            messages=trimmed_messages,
            step=step,
            bytes_removed=max(0, total_chars_before - total_chars_after),
            tokens_removed=max(0, current_tokens - tokens_after),
        )

    return None


LADDER_ORDER: tuple[DegradationStepKind, ...] = DEFAULT_CONTEXT_DEGRADATION_LADDER_ORDER


def apply_degradation_ladder(
    *,
    messages: list[ChatMessage],
    candidates: Sequence[ContextCandidate],
    budget_tokens: int,
    prefer_longterm_memory: bool,
    prefer_rag_when_enabled: bool,
    count_tokens: Callable[[str], int],
    policy_ladder_order: tuple[DegradationStepKind, ...],
) -> tuple[list[ChatMessage], tuple[str, ...]]:
    """Run compiler degradation steps until within budget or ladder exhausted."""
    from intergrax.runtime.nexus.context.context_compiler import classify_candidates

    working = list(messages)
    applied: list[str] = []

    for step in policy_ladder_order:
        if step == DegradationStepKind.FULL:
            continue
        effective_step = step
        if step == DegradationStepKind.REDUCE_INJECTION_BLOCKS:
            effective_step = DegradationStepKind.DROP_LOWEST_SCORED

        candidates = classify_candidates(working, count_tokens=count_tokens)
        if _sum_tokens(candidates) <= budget_tokens:
            break

        result = apply_degradation_step(
            messages=working,
            candidates=candidates,
            step=effective_step,
            budget_tokens=budget_tokens,
            prefer_longterm_memory=prefer_longterm_memory,
            prefer_rag_when_enabled=prefer_rag_when_enabled,
            count_tokens=count_tokens,
        )
        if result is None:
            continue
        working = result.messages
        applied.append(result.step.value)
        candidates = classify_candidates(working, count_tokens=count_tokens)
        if _sum_tokens(candidates) <= budget_tokens:
            break

    return working, tuple(applied)
