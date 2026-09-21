# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Ports for post-qualification resume — AW depends on abstractions only (UCA-6C)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    WorkerQualifiedCapabilityExecutionRequest,
    WorkerQualifiedCapabilityExecutionResult,
)
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    QualifiedCapabilityBindingRequest,
    QualifiedCapabilityBindingResult,
)


@runtime_checkable
class QualifiedCapabilityBindingPort(Protocol):
    """Pluginable qualified-subject runtime binding."""

    def bind(
        self,
        request: QualifiedCapabilityBindingRequest,
    ) -> QualifiedCapabilityBindingResult: ...


@runtime_checkable
class WorkerQualifiedCapabilityExecutionPort(Protocol):
    """Sole production execution authority for qualified capabilities."""

    def execute(
        self,
        request: WorkerQualifiedCapabilityExecutionRequest,
    ) -> WorkerQualifiedCapabilityExecutionResult: ...


__all__ = [
    "QualifiedCapabilityBindingPort",
    "WorkerQualifiedCapabilityExecutionPort",
]
