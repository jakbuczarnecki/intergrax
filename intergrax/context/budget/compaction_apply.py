# © Artur Czarnecki. All rights reserved.

"""Canonical fragment compaction phase (planner-owned semantic window)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from intergrax.context.budget.compaction import (
    ContextCompactionInput,
    ContextCompactionProvenance,
    ContextCompactionStrategy,
)
from intergrax.context.budget.compaction_validation import validate_compaction_result
from intergrax.context.contracts import ContextFragment
from intergrax.context.errors import ContextProviderContractViolationError


def apply_fragment_compaction(
    fragments: list[ContextFragment],
    *,
    strategy: ContextCompactionStrategy,
    count_text: Callable[[str], int],
    fragment_budget_tokens: int,
) -> tuple[list[ContextFragment], tuple[ContextCompactionProvenance, ...]]:
    """Compact optional fragments when ranked allocation exceeds fragment budget."""
    if not fragments:
        return fragments, ()

    total = sum(max(0, count_text(fragment.content or "")) for fragment in fragments)
    if total <= fragment_budget_tokens:
        return fragments, ()

    working = list(fragments)
    provenance_records: list[ContextCompactionProvenance] = []
    optional_indices = [
        index
        for index, fragment in enumerate(working)
        if not fragment.mandatory
    ]
    optional_indices.sort(
        key=lambda index: (
            working[index].relevance_score,
            working[index].fragment_id,
        ),
    )

    for index in optional_indices:
        current_total = sum(max(0, count_text(item.content or "")) for item in working)
        if current_total <= fragment_budget_tokens:
            break
        fragment = working[index]
        per_fragment_budget = max(1, fragment_budget_tokens // max(1, len(working)))
        try:
            outcome = strategy.compact(
                ContextCompactionInput(
                    fragment=fragment,
                    target_token_budget=per_fragment_budget,
                ),
            )
        except Exception as exc:
            raise ContextProviderContractViolationError(
                f"compaction.strategy_failed:{strategy.strategy_id}",
            ) from exc
        if outcome is None:
            continue
        validate_compaction_result(fragment, outcome)
        provenance_records.append(outcome.provenance)
        working[index] = outcome.fragment
        if outcome.fragment.token_estimate <= 0:
            token_estimate = max(1, count_text(outcome.fragment.content or ""))
            working[index] = replace(outcome.fragment, token_estimate=token_estimate)

    return working, tuple(provenance_records)
