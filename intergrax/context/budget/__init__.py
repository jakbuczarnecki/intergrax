# © Artur Czarnecki. All rights reserved.

"""Context budgeting, compaction, and degradation contracts (CE-02)."""

from intergrax.context.budget.compaction import (
    ContextCompactionInput,
    ContextCompactionProvenance,
    ContextCompactionResult,
    ContextCompactionStrategy,
    DeterministicTailCompactionStrategy,
    NoOpContextCompactionStrategy,
)
from intergrax.context.budget.contracts import (
    ContextBudgetResolveInput,
    ContextBudgetUnsatisfiableError,
    ModelContextCapabilitySnapshot,
    ResolvedModelContextBudget,
)
from intergrax.context.budget.default_model_budget_policy import DefaultContextModelBudgetPolicy
from intergrax.context.budget.degradation import (
    ContextDegradationPolicy,
    DefaultContextDegradationPolicy,
)
from intergrax.context.budget.model_budget_policy import ContextModelBudgetPolicy
from intergrax.context.budget.resolver import global_allocatable_tokens, resolve_authoritative_model_budget
from intergrax.context.budget.token_counter import CharEstimateContextTokenCounter, ContextTokenCounter

__all__ = [
    "CharEstimateContextTokenCounter",
    "ContextBudgetResolveInput",
    "ContextBudgetUnsatisfiableError",
    "ContextCompactionInput",
    "ContextCompactionProvenance",
    "ContextCompactionResult",
    "ContextCompactionStrategy",
    "ContextDegradationPolicy",
    "ContextModelBudgetPolicy",
    "ContextTokenCounter",
    "DefaultContextDegradationPolicy",
    "DefaultContextModelBudgetPolicy",
    "DeterministicTailCompactionStrategy",
    "ModelContextCapabilitySnapshot",
    "NoOpContextCompactionStrategy",
    "ResolvedModelContextBudget",
    "global_allocatable_tokens",
    "resolve_authoritative_model_budget",
]
