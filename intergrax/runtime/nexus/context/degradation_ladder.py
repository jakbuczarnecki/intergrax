# © Artur Czarnecki. All rights reserved.

"""Ordered context degradation ladder (Phase MEM-DEPTH-1.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.context.context_budget import (
    ContextBudgetPolicy,
    trim_message_to_budget_tokenizer_aware,
)
from intergrax.runtime.nexus.context.context_compiler_models import (
    ContextCandidate,
    ContextCandidateSource,
    DegradationStepKind,
)


@dataclass(frozen=True, slots=True)
class DegradationApplyResult:
    messages: List[ChatMessage]
    step: DegradationStepKind
    bytes_removed: int
    tokens_removed: int


def _sum_tokens(candidates: Sequence[ContextCandidate]) -> int:
    return sum(candidate.token_estimate for candidate in candidates)


def _rebuild_messages(
    messages: List[ChatMessage],
    keep_indices: set[int],
) -> List[ChatMessage]:
    return [message for index, message in enumerate(messages) if index in keep_indices]


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
    messages: List[ChatMessage],
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
        history_indices = [
            candidate.message_index
            for candidate in candidates
            if candidate.source == ContextCandidateSource.SESSION_HISTORY
        ]
        if not history_indices:
            return None
        drop_index = min(history_indices)
        keep = {candidate.message_index for candidate in candidates if candidate.message_index != drop_index}
        new_messages = _rebuild_messages(messages, keep)
        removed_chars = len(messages[drop_index].content or "")
        new_candidates = [c for c in candidates if c.message_index in keep]
        return DegradationApplyResult(
            messages=new_messages,
            step=step,
            bytes_removed=removed_chars,
            tokens_removed=max(0, current_tokens - _sum_tokens(new_candidates)),
        )

    if step == DegradationStepKind.DROP_LOWEST_SCORED:
        droppable = [
            candidate
            for candidate in candidates
            if not candidate.mandatory
        ]
        if not droppable:
            return None
        drop = min(droppable, key=lambda c: c.score)
        keep = {candidate.message_index for candidate in candidates if candidate.message_index != drop.message_index}
        new_messages = _rebuild_messages(messages, keep)
        removed_chars = len(messages[drop.message_index].content or "")
        new_candidates = [c for c in candidates if c.message_index in keep]
        return DegradationApplyResult(
            messages=new_messages,
            step=step,
            bytes_removed=removed_chars,
            tokens_removed=max(0, current_tokens - _sum_tokens(new_candidates)),
        )

    if step == DegradationStepKind.TOKENIZER_HARD_TRIM:
        policy = ContextBudgetPolicy(max_chars=budget_tokens * 4, max_tokens_estimate=budget_tokens)
        total_chars_before = sum(len(m.content or "") for m in messages)
        trimmed_messages: List[ChatMessage] = []
        for message in messages:
            if message.role == "user" and message is messages[-1]:
                trimmed_messages.append(message)
                continue
            result = trim_message_to_budget_tokenizer_aware(
                message.content or "",
                policy,
                count_tokens=count_tokens,
            )
            trimmed_messages.append(
                ChatMessage(role=message.role, content=result.message, metadata=message.metadata)
            )
        total_chars_after = sum(len(m.content or "") for m in trimmed_messages)
        return DegradationApplyResult(
            messages=trimmed_messages,
            step=step,
            bytes_removed=max(0, total_chars_before - total_chars_after),
            tokens_removed=max(0, current_tokens - budget_tokens),
        )

    return None


LADDER_ORDER: tuple[DegradationStepKind, ...] = (
    DegradationStepKind.FULL,
    DegradationStepKind.DROP_OPTIONAL_INJECTIONS,
    DegradationStepKind.REDUCE_INJECTION_BLOCKS,
    DegradationStepKind.TRUNCATE_OLDEST_HISTORY,
    DegradationStepKind.DROP_LOWEST_SCORED,
    DegradationStepKind.TOKENIZER_HARD_TRIM,
)
