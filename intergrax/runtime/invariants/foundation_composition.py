# © Artur Czarnecki. All rights reserved.

"""Composition seam for RI-01 foundation packs (imports domain adapters)."""

from __future__ import annotations

from intergrax.contracts.runtime_invariants import RuntimeInvariantRulePack
from intergrax.runtime.execution.delegated_execution.invariants import (
    DefaultDelegatedProviderInvariantProbe,
    DelegatedProviderRuntimeInvariantRulePack,
)
from intergrax.runtime.execution.invariants import (
    DefaultExecutionInvariantProbe,
    ExecutionRuntimeInvariantRulePack,
)
from intergrax.runtime.governance.invariants import (
    DefaultGovernanceInvariantProbe,
    GovernanceRuntimeInvariantRulePack,
)
from intergrax.runtime.invariants.clock import SystemRuntimeInvariantEvaluationClock
from intergrax.runtime.invariants.evaluation_id import DefaultRuntimeInvariantEvaluationIdFactory
from intergrax.runtime.invariants.service import RuntimeInvariantService


def foundation_runtime_invariant_rule_packs() -> tuple[RuntimeInvariantRulePack, ...]:
    """Default three domain packs for qualification and diagnostics consumers."""
    return (
        ExecutionRuntimeInvariantRulePack(DefaultExecutionInvariantProbe()),
        DelegatedProviderRuntimeInvariantRulePack(DefaultDelegatedProviderInvariantProbe()),
        GovernanceRuntimeInvariantRulePack(DefaultGovernanceInvariantProbe()),
    )


def compose_foundation_runtime_invariant_service() -> RuntimeInvariantService:
    return RuntimeInvariantService(
        rule_packs=foundation_runtime_invariant_rule_packs(),
        clock=SystemRuntimeInvariantEvaluationClock(),
        evaluation_id_factory=DefaultRuntimeInvariantEvaluationIdFactory(),
    )


__all__ = [
    "compose_foundation_runtime_invariant_service",
    "foundation_runtime_invariant_rule_packs",
]
