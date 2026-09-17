# © Artur Czarnecki. All rights reserved.

"""Replaceable global budget resolution policy (CE-02)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.context.budget.contracts import ContextBudgetResolveInput, ResolvedModelContextBudget


@runtime_checkable
class ContextModelBudgetPolicy(Protocol):
    """Resolves the single authoritative model-facing input budget."""

    @property
    def policy_id(self) -> str: ...

    @property
    def policy_version(self) -> str: ...

    def resolve_budget(self, inputs: ContextBudgetResolveInput) -> ResolvedModelContextBudget: ...
