# © Artur Czarnecki. All rights reserved.

"""Budget reaction hook protocol (architecture §25.5.3 · ACP-TOK-3)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BudgetReactionHook(Protocol):
    """Optional Tier-3 host callback for budget threshold / exceed events."""

    async def on_budget_threshold(self, payload: dict[str, Any]) -> None: ...

    async def on_budget_exceeded(self, payload: dict[str, Any]) -> None: ...


@runtime_checkable
class CustomBudgetReactionHook(Protocol):
    """Host callback for ``BudgetExceededReaction.CUSTOM_HOOK`` (non-standard hook surface)."""

    async def on_custom_budget_hook(self, hook_id: str, payload: dict[str, Any]) -> None: ...
