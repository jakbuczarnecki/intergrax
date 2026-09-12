# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Autonomy ↔ execution spine integration port (SELF-HEALING R6.3)."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from intergrax.contracts.self_healing.autonomy.execution_authorization import AutonomyExecutionAuthorization
from intergrax.contracts.self_healing.autonomy.guard import AutonomyExecutionAdmissionContext

RequestT = TypeVar("RequestT", contravariant=True)


@runtime_checkable
class AutonomyAdmissionContextSource(Protocol[RequestT]):
    """Extract optional autonomy admission from a spine request — ``None`` preserves legacy flow."""

    def admission_for(self, request: RequestT) -> AutonomyExecutionAdmissionContext | None: ...


@runtime_checkable
class AutonomyExecutionBoundary(Protocol):
    """
    Sole integration surface between autonomy controls and the execution spine.

    Must not import or invoke executors.
    """

    def authorize(
        self,
        admission: AutonomyExecutionAdmissionContext,
    ) -> AutonomyExecutionAuthorization: ...

    def spine_admission_hook(
        self,
        source: AutonomyAdmissionContextSource[RequestT],
    ) -> object:
        """Return an ``ExecutionAdmissionHook`` compatible object for ``ExecutionBoundary``."""
        ...


__all__ = [
    "AutonomyAdmissionContextSource",
    "AutonomyExecutionBoundary",
]
