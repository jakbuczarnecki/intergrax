# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit composition for meaningful side-effect authorization (GR-3-R1).

Default inner guard binding belongs here — not inside policy consumers.
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


def build_default_canonical_inner_execution_guard(
    *,
    task_scope: ActiveExecutionTaskScopePort | None = None,
) -> CanonicalInnerExecutionGuardPort:
    """Platform default inner guard — explicit composition root only."""
    return DefaultCanonicalInnerExecutionGuard(task_scope=task_scope)


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
    return build_meaningful_side_effect_authorization_boundary(
        enforcement_gate=enforcement_gate,
        inner_execution_guard=build_default_canonical_inner_execution_guard(
            task_scope=task_scope,
        ),
    )


__all__ = [
    "build_default_canonical_inner_execution_guard",
    "build_default_wired_meaningful_side_effect_authorization_boundary",
    "build_meaningful_side_effect_authorization_boundary",
]
