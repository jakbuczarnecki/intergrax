# © Artur Czarnecki. All rights reserved.

"""Default fragment budget allocator (MEM-XINT-5)."""

from __future__ import annotations

from intergrax.context.contracts import (
    BudgetAllocationResult,
    ContextAssemblyRequest,
    ContextFragment,
    ContextFragmentSource,
    ContextPolicyReasonCode,
)


class DefaultContextBudgetAllocator:
    """Greedy token budget over ranked fragments — preserves mandatory entries."""

    _PROTECTED_SOURCES = frozenset(
        {
            ContextFragmentSource.SESSION_HISTORY,
            ContextFragmentSource.SYSTEM_INSTRUCTIONS,
            ContextFragmentSource.POLICY_OVERLAY,
        }
    )

    @property
    def strategy_id(self) -> str:
        return "default_context_budget_allocator.v1"

    def allocate(
        self,
        fragments: list[ContextFragment],
        budget_tokens: int,
        request: ContextAssemblyRequest,
    ) -> BudgetAllocationResult:
        _ = request
        if budget_tokens < 0:
            raise ValueError("budget_tokens must be >= 0")
        included: list[ContextFragment] = []
        excluded: list[tuple[ContextFragment, str]] = []
        used = 0
        for fragment in fragments:
            cost = max(fragment.token_estimate, 1)
            protected = fragment.mandatory or fragment.source in self._PROTECTED_SOURCES
            if protected:
                included.append(fragment)
                used += cost
                continue
            if used + cost <= budget_tokens:
                included.append(fragment)
                used += cost
                continue
            excluded.append((fragment, ContextPolicyReasonCode.BUDGET_EXCLUDED.value))
        return BudgetAllocationResult(
            included=tuple(included),
            excluded=tuple(excluded),
            total_tokens=used,
            budget_tokens=budget_tokens,
        )
