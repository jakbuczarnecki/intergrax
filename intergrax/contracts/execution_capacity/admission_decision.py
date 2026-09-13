# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""EE-B1.2 — typed root execution capacity assessment (decision plane only).

Orthogonal to :class:`~intergrax.contracts.execution_capacity_admission.ExecutionCapacityAdmissionPort`
acquire/release lifecycle. Assessment uses **platform-authoritative** counters only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.execution_capacity_admission import (
    ExecutionCapacityOverloadMode,
)


class ExecutionCapacityAdmissionDecision(StrEnum):
    """Pre-admission capacity outcome (no AttemptId / execution lifecycle side effects)."""

    ALLOW = "ALLOW"
    DEFER = "DEFER"
    REJECT = "REJECT"


@dataclass(frozen=True, slots=True)
class ExecutionCapacityAssessmentContext:
    """Immutable, provider-neutral capacity snapshot for one assessment."""

    active_root_executions: int
    capacity_limit: int
    overload_mode: ExecutionCapacityOverloadMode

    def __post_init__(self) -> None:
        if self.active_root_executions < 0:
            raise ValueError("active_root_executions must be >= 0")
        if self.capacity_limit < 1:
            raise ValueError("capacity_limit must be >= 1")


@runtime_checkable
class ExecutionCapacityEvaluator(Protocol):
    """Pluginable capacity decision (local, distributed, tenant quota, etc.)."""

    def evaluate(
        self,
        context: ExecutionCapacityAssessmentContext,
    ) -> ExecutionCapacityAdmissionDecision:
        """Return typed admission decision without reserving a slot."""
        ...


def assess_root_execution_capacity(
    context: ExecutionCapacityAssessmentContext,
) -> ExecutionCapacityAdmissionDecision:
    """Deterministic local assessment for process-local root slot policy.

    * **ALLOW** — slot available now.
    * **REJECT** — saturated under ``REJECT`` overload mode (fail-fast backpressure).
    * **DEFER** — saturated under ``WAIT_WITH_TIMEOUT`` (caller/port must bound wait;
      no hidden queue in this function).
    """
    if type(context) is not ExecutionCapacityAssessmentContext:
        raise TypeError("context must be ExecutionCapacityAssessmentContext")
    if context.active_root_executions < context.capacity_limit:
        return ExecutionCapacityAdmissionDecision.ALLOW
    if context.overload_mode is ExecutionCapacityOverloadMode.REJECT:
        return ExecutionCapacityAdmissionDecision.REJECT
    return ExecutionCapacityAdmissionDecision.DEFER


class RootExecutionCapacityEvaluator:
    """Default EE-B1.2 evaluator delegating to :func:`assess_root_execution_capacity`."""

    __slots__ = ()

    def evaluate(
        self,
        context: ExecutionCapacityAssessmentContext,
    ) -> ExecutionCapacityAdmissionDecision:
        return assess_root_execution_capacity(context)
