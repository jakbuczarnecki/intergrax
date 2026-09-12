# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Guard that exposes full execution authorization (SELF-HEALING R6.3)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.autonomy.execution_authorization import AutonomyExecutionAuthorization
from intergrax.contracts.self_healing.autonomy.guard import (
    AutonomyExecutionAdmissionContext,
    AutonomyExecutionGuard,
)


@runtime_checkable
class AutonomyExecutionAuthorizingGuard(AutonomyExecutionGuard, Protocol):
    def authorize(
        self,
        admission: AutonomyExecutionAdmissionContext,
    ) -> AutonomyExecutionAuthorization:
        """Produce auditable authorization — must not invoke executors."""
        ...


__all__ = ["AutonomyExecutionAuthorizingGuard"]
