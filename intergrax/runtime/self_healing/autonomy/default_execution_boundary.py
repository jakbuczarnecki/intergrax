# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Autonomy execution boundary adapter for the canonical execution spine (R6.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from intergrax.contracts.self_healing.autonomy.execution_authorization import AutonomyExecutionAuthorization
from intergrax.contracts.self_healing.autonomy.execution_boundary import AutonomyAdmissionContextSource
from intergrax.contracts.self_healing.autonomy.execution_denied import AutonomyExecutionDeniedError
from intergrax.contracts.self_healing.autonomy.authorizing_guard import AutonomyExecutionAuthorizingGuard
from intergrax.contracts.self_healing.autonomy.guard import AutonomyExecutionAdmissionContext
from intergrax.runtime.execution.boundary import ExecutionAdmissionHook

RequestT = TypeVar("RequestT")


@dataclass(frozen=True, slots=True)
class _AutonomySpineAdmissionHook(Generic[RequestT]):
    guard: AutonomyExecutionAuthorizingGuard
    source: AutonomyAdmissionContextSource[RequestT]

    async def admit(self, request: RequestT) -> None:
        admission = self.source.admission_for(request)
        if admission is None:
            return
        authorization = self.guard.authorize(admission)
        if not authorization.permits_execution():
            raise AutonomyExecutionDeniedError(authorization)


@dataclass(frozen=True, slots=True)
class DefaultAutonomyExecutionBoundary:
    guard: AutonomyExecutionAuthorizingGuard

    def authorize(
        self,
        admission: AutonomyExecutionAdmissionContext,
    ) -> AutonomyExecutionAuthorization:
        return self.guard.authorize(admission)

    def spine_admission_hook(
        self,
        source: AutonomyAdmissionContextSource[RequestT],
    ) -> ExecutionAdmissionHook[RequestT]:
        return _AutonomySpineAdmissionHook(self.guard, source)


__all__ = ["DefaultAutonomyExecutionBoundary"]
