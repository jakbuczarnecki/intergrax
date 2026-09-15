# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Default GR-3 inner execution guard — active ContextVar + task scope port."""

from __future__ import annotations

from intergrax.contracts.active_execution_task_scope import ActiveExecutionTaskScopePort
from intergrax.contracts.canonical_inner_governance import (
    CanonicalInnerExecutionGuardPort,
    require_active_execution_for_meaningful_side_effect,
)
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistryTaskScopeResolver


class DefaultCanonicalInnerExecutionGuard(CanonicalInnerExecutionGuardPort):
    """Platform default — no global registry; uses in-flight task registration."""

    __slots__ = ("_task_scope",)

    def __init__(
        self,
        *,
        task_scope: ActiveExecutionTaskScopePort | None = None,
    ) -> None:
        self._task_scope = task_scope or ActiveTaskRegistryTaskScopeResolver()

    def assert_meaningful_side_effect_bound(
        self,
        request: MeaningfulSideEffectRequest,
    ) -> None:
        require_active_execution_for_meaningful_side_effect(
            request,
            task_scope=self._task_scope,
        )


__all__ = ["DefaultCanonicalInnerExecutionGuard"]
