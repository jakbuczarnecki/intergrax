# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Context Engineering plugin protocols (Phase CE-1.3)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.context.contracts import (
    AssembledContext,
    BudgetAllocationResult,
    ContextAssemblyRequest,
    ContextFragment,
    ContextFragmentSource,
    ContextProviderContext,
)
from intergrax.context.session_history import SessionHistorySnapshot
from intergrax.llm.messages import ChatMessage


@runtime_checkable
class ContextSourceProvider(Protocol):
    """Collects candidate fragments for one assembly request."""

    @property
    def provider_id(self) -> str: ...

    @property
    def supported_sources(self) -> frozenset[ContextFragmentSource]: ...

    async def collect(
        self,
        request: ContextAssemblyRequest,
        ctx: ContextProviderContext,
    ) -> list[ContextFragment]: ...


@runtime_checkable
class ContextRanker(Protocol):
    """Orders fragments after collect and before budget allocation."""

    @property
    def ranker_id(self) -> str: ...

    def rank(
        self,
        fragments: list[ContextFragment],
        request: ContextAssemblyRequest,
    ) -> list[ContextFragment]: ...


@runtime_checkable
class ContextBudgetAllocator(Protocol):
    """Allocates token budget across ranked fragments."""

    def allocate(
        self,
        fragments: list[ContextFragment],
        budget_tokens: int,
        request: ContextAssemblyRequest,
    ) -> BudgetAllocationResult: ...


@runtime_checkable
class ContextFormatter(Protocol):
    """Formats allocated fragments into chat messages."""

    def format(
        self,
        fragments: list[ContextFragment],
        request: ContextAssemblyRequest,
    ) -> list[ChatMessage]: ...


@runtime_checkable
class ContextValidationResult(Protocol):
    valid: bool
    errors: tuple[str, ...]


@runtime_checkable
class ContextValidator(Protocol):
    """Validates assembled context before LLM invocation."""

    def validate(
        self,
        assembled: AssembledContext,
        request: ContextAssemblyRequest,
    ) -> ContextValidationResult: ...


@runtime_checkable
class ContextEngine(Protocol):
    """Unified context assembly entry (CE-3)."""

    @property
    def engine_id(self) -> str: ...

    async def assemble(self, request: ContextAssemblyRequest) -> AssembledContext: ...


@runtime_checkable
class SessionHistoryProvider(Protocol):
    """Loads structured session history for planning and collection."""

    async def load_snapshot(
        self,
        request: ContextAssemblyRequest,
        ctx: ContextProviderContext,
    ) -> SessionHistorySnapshot | None: ...
