# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Generic pre-EE qualified capability execution intent preparation (S24-GAP-02-P3)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.autonomous_work.capability_acquisition import WorkerCapabilityNeed
from intergrax.contracts.capability_qualification.qualification_result import (
    CapabilityQualificationResult,
)
from intergrax.contracts.execution_identity import TaskId


class QualifiedCapabilityExecutionIntentPreparationOutcome(StrEnum):
    """Typed preparation terminal — no boolean-only API."""

    NOT_APPLICABLE = "not_applicable"
    CREATED = "created"
    ALREADY_RECORDED_IDENTICAL = "already_recorded_identical"
    UNAVAILABLE = "unavailable"
    CONFLICT = "conflict"
    INTEGRITY_FAILURE = "integrity_failure"
    INVALID_OPERATION = "invalid_operation"


@dataclass(frozen=True, slots=True)
class QualifiedCapabilityExecutionIntentPreparationRequest:
    """Minimal AW→domain seam — no full fulfillment request."""

    need: WorkerCapabilityNeed
    qualification_result: CapabilityQualificationResult
    execution_request_id: str
    binding_operation_id: str
    resume_operation_id: str
    worker_need_id: str
    tenant_id: str
    task_id: TaskId


@dataclass(frozen=True, slots=True)
class QualifiedCapabilityExecutionIntentPreparationResult:
    outcome: QualifiedCapabilityExecutionIntentPreparationOutcome
    reason_detail: str = ""


@runtime_checkable
class QualifiedCapabilityExecutionIntentPreparationPort(Protocol):
    """Domain-owned durable intent preparation — AW must not know Marketplace."""

    def prepare(
        self,
        request: QualifiedCapabilityExecutionIntentPreparationRequest,
    ) -> QualifiedCapabilityExecutionIntentPreparationResult: ...


__all__ = [
    "QualifiedCapabilityExecutionIntentPreparationOutcome",
    "QualifiedCapabilityExecutionIntentPreparationPort",
    "QualifiedCapabilityExecutionIntentPreparationRequest",
    "QualifiedCapabilityExecutionIntentPreparationResult",
]
