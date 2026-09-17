# © Artur Czarnecki. All rights reserved.

"""Typed context budget contracts (CE-02)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from intergrax.context.contracts import ContextAssemblyRequest, ContextBudgetSnapshot


@dataclass(frozen=True, slots=True)
class ModelContextCapabilitySnapshot:
    """Provider-neutral model window snapshot (no vendor types in CE core)."""

    model_context_window: int
    reserved_output_tokens: int
    platform_margin_tokens: int

    def __post_init__(self) -> None:
        if self.model_context_window < 1:
            raise ValueError("model_context_window must be >= 1")
        if self.reserved_output_tokens < 0:
            raise ValueError("reserved_output_tokens must be >= 0")
        if self.platform_margin_tokens < 0:
            raise ValueError("platform_margin_tokens must be >= 0")
        reserved = self.reserved_output_tokens + self.platform_margin_tokens
        if reserved >= self.model_context_window:
            raise ValueError("reserved tokens must fit inside model_context_window")

    @property
    def available_input_tokens(self) -> int:
        return max(
            1,
            self.model_context_window
            - self.reserved_output_tokens
            - self.platform_margin_tokens,
        )


@dataclass(frozen=True, slots=True)
class ResolvedModelContextBudget:
    """Authoritative global model-facing budget after policy resolution."""

    model_context_window: int
    reserved_output_tokens: int
    platform_margin_tokens: int
    available_input_tokens: int
    mandatory_reserve_tokens: int
    allocatable_tokens: int
    request_cap_tokens: int | None
    policy_id: str
    policy_version: str

    def __post_init__(self) -> None:
        if self.available_input_tokens < 1:
            raise ValueError("available_input_tokens must be >= 1")
        if self.allocatable_tokens < 0:
            raise ValueError("allocatable_tokens must be >= 0")
        if self.mandatory_reserve_tokens < 0:
            raise ValueError("mandatory_reserve_tokens must be >= 0")


@dataclass(frozen=True, slots=True)
class ContextBudgetResolveInput:
    """Inputs for ``ContextModelBudgetPolicy.resolve_budget``."""

    capability: ModelContextCapabilitySnapshot
    request_budget: ContextBudgetSnapshot
    mandatory_reserve_tokens: int = 0


class ContextBudgetUnsatisfiableError(ValueError):
    """Mandatory content cannot fit in available input budget (fail-closed)."""

    reason_code: str = "budget.unsatisfiable.mandatory_overflow"

    def __init__(self, *, detail: str = "", mandatory_tokens: int = 0, available_tokens: int = 0) -> None:
        self.mandatory_tokens = mandatory_tokens
        self.available_tokens = available_tokens
        message = self.reason_code
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)
