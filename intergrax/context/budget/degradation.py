# © Artur Czarnecki. All rights reserved.

"""Replaceable degradation ladder policy (CE-02)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.context.budget.contracts import (
    DEFAULT_CONTEXT_DEGRADATION_LADDER_ORDER,
    DegradationStepKind,
)


@runtime_checkable
class ContextDegradationPolicy(Protocol):
    @property
    def policy_id(self) -> str: ...

    def ladder_order(self) -> tuple[DegradationStepKind, ...]: ...


class DefaultContextDegradationPolicy:
    """Shipped ladder order (canonical ``DEFAULT_CONTEXT_DEGRADATION_LADDER_ORDER``)."""

    @property
    def policy_id(self) -> str:
        return "default_context_degradation_policy.v1"

    def ladder_order(self) -> tuple[DegradationStepKind, ...]:
        return DEFAULT_CONTEXT_DEGRADATION_LADDER_ORDER
