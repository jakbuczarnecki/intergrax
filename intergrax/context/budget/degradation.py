# © Artur Czarnecki. All rights reserved.

"""Replaceable degradation ladder policy (CE-02)."""

from __future__ import annotations

from typing import Callable, Protocol, Sequence, runtime_checkable

from intergrax.context.budget.contracts import DegradationStepKind
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.context.context_compiler_models import ContextCandidate


@runtime_checkable
class ContextDegradationPolicy(Protocol):
    @property
    def policy_id(self) -> str: ...

    def ladder_order(self) -> tuple[DegradationStepKind, ...]: ...


class DefaultContextDegradationPolicy:
    """Shipped ladder — mirrors legacy ``LADDER_ORDER``."""

    @property
    def policy_id(self) -> str:
        return "default_context_degradation_policy.v1"

    def ladder_order(self) -> tuple[DegradationStepKind, ...]:
        from intergrax.runtime.nexus.context.degradation_ladder import LADDER_ORDER

        return LADDER_ORDER


def apply_degradation_ladder(
    *,
    messages: list[ChatMessage],
    candidates: Sequence[ContextCandidate],
    budget_tokens: int,
    prefer_longterm_memory: bool,
    prefer_rag_when_enabled: bool,
    count_tokens: Callable[[str], int],
    policy: ContextDegradationPolicy,
) -> tuple[list[ChatMessage], tuple[str, ...]]:
    """Run degradation steps until within budget or ladder exhausted."""
    from intergrax.runtime.nexus.context.degradation_ladder import apply_degradation_step

    working = list(messages)
    applied: list[str] = []

    for step in policy.ladder_order():
        if step == DegradationStepKind.FULL:
            continue
        effective_step = step
        if step == DegradationStepKind.REDUCE_INJECTION_BLOCKS:
            effective_step = DegradationStepKind.DROP_LOWEST_SCORED

        candidates = _reclassify(working, count_tokens)
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
        candidates = _reclassify(working, count_tokens)
        if _sum_tokens(candidates) <= budget_tokens:
            break

    return working, tuple(applied)


def _sum_tokens(candidates: Sequence[ContextCandidate]) -> int:
    return sum(candidate.token_estimate for candidate in candidates)


def _reclassify(
    messages: list[ChatMessage],
    count_tokens: Callable[[str], int],
) -> list[ContextCandidate]:
    from intergrax.runtime.nexus.context.context_compiler import classify_candidates

    return classify_candidates(messages, count_tokens=count_tokens)
