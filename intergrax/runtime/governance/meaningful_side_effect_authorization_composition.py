# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit composition for meaningful side-effect authorization (GR-3-R1 / GR-3-R2).

Default inner guard and task-scope binding belong here — not inside policy consumers.
"""

from __future__ import annotations

from intergrax.collaborative_work.enforcement_gate import CollaborativeWorkEnforcementGate
from intergrax.contracts.active_execution_task_scope import ActiveExecutionTaskScopePort
from intergrax.contracts.canonical_inner_governance import CanonicalInnerExecutionGuardPort
from intergrax.runtime.governance.canonical_inner_execution_guard import (
    DefaultCanonicalInnerExecutionGuard,
)
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
)
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistryTaskScopeResolver


def build_canonical_inner_execution_guard(
    *,
    task_scope: ActiveExecutionTaskScopePort,
) -> CanonicalInnerExecutionGuardPort:
    """Wire explicit task-scope implementation into the platform default inner guard."""
    return DefaultCanonicalInnerExecutionGuard(task_scope=task_scope)


def build_default_canonical_inner_execution_guard() -> CanonicalInnerExecutionGuardPort:
    """Platform default inner guard — explicit composition root only."""
    return build_canonical_inner_execution_guard(
        task_scope=ActiveTaskRegistryTaskScopeResolver(),
    )


def build_meaningful_side_effect_authorization_boundary(
    *,
    enforcement_gate: CollaborativeWorkEnforcementGate,
    inner_execution_guard: CanonicalInnerExecutionGuardPort,
) -> MeaningfulSideEffectAuthorizationBoundary:
    """Wire enforcement gate and inner guard behind the shared policy boundary."""
    return MeaningfulSideEffectAuthorizationBoundary(
        enforcement_gate=enforcement_gate,
        inner_execution_guard=inner_execution_guard,
    )


def build_default_wired_meaningful_side_effect_authorization_boundary(
    *,
    enforcement_gate: CollaborativeWorkEnforcementGate,
    task_scope: ActiveExecutionTaskScopePort | None = None,
) -> MeaningfulSideEffectAuthorizationBoundary:
    """Production-default wiring — constructs platform default inner guard explicitly."""
    resolved_task_scope = (
        task_scope
        if task_scope is not None
        else ActiveTaskRegistryTaskScopeResolver()
    )
    return build_meaningful_side_effect_authorization_boundary(
        enforcement_gate=enforcement_gate,
        inner_execution_guard=build_canonical_inner_execution_guard(
            task_scope=resolved_task_scope,
        ),
    )


__all__ = [
    "build_canonical_inner_execution_guard",
    "build_default_canonical_inner_execution_guard",
    "build_default_wired_meaningful_side_effect_authorization_boundary",
    "build_meaningful_side_effect_authorization_boundary",
]
