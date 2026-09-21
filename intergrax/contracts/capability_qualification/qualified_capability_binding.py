# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Qualified capability runtime binding contracts — pluginable, AW-opaque (UCA-6C)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_qualification.qualified_subject import (
    QualifiedCapabilitySubject,
)
from intergrax.contracts.capability_qualification.qualification_result import (
    CapabilityQualificationResult,
)
from intergrax.contracts.execution_identity import TaskId, validate_task_id

SCHEMA_QUALIFIED_CAPABILITY_BINDING_REQUEST_V1: Final = (
    "qualified_capability_binding_request.v1"
)
SCHEMA_QUALIFIED_CAPABILITY_BINDING_RESULT_V1: Final = (
    "qualified_capability_binding_result.v1"
)
SCHEMA_QUALIFIED_CAPABILITY_EXECUTION_TARGET_V1: Final = (
    "qualified_capability_execution_target.v1"
)
_NON_EMPTY = Field(min_length=1)


class QualifiedCapabilityBindingOutcome(StrEnum):
    """Binding coordination outcome — not execution or host availability."""

    BOUND = "bound"
    BLOCKED = "blocked"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    REQUIRES_HITL = "requires_hitl"
    CONFLICT = "conflict"
    NOT_SUPPORTED = "not_supported"
    NO_PROVIDER = "no_provider"


class QualifiedCapabilityBindingReasonCode(StrEnum):
    """Stable binding reason codes."""

    NONE = "none"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    SUBJECT_NOT_SUPPORTED = "subject_not_supported"
    SUBJECT_MISMATCH = "subject_mismatch"
    POLICY_BLOCKED = "policy_blocked"
    INTEGRITY_CONFLICT = "integrity_conflict"
    HITL_REQUIRED = "hitl_required"
    INTERNAL_ERROR = "internal_error"


def derive_qualified_capability_binding_operation_id(
    *,
    resume_operation_id: str,
    qualified_subject_reference: str,
) -> str:
    resume_id = require_non_empty_text(resume_operation_id, label="resume_operation_id")
    subject_ref = require_non_empty_text(
        qualified_subject_reference,
        label="qualified_subject_reference",
    )
    return f"qualified-capability-binding:{resume_id}:{subject_ref}"


class QualifiedCapabilityExecutionTarget(BaseModel):
    """Typed execution handoff reference — consumed only by Execution Engine ports."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["qualified_capability_execution_target.v1"] = (
        SCHEMA_QUALIFIED_CAPABILITY_EXECUTION_TARGET_V1
    )
    execution_target_reference: str = _NON_EMPTY
    binding_provider_id: str = _NON_EMPTY
    qualified_subject_reference: str = _NON_EMPTY

    @model_validator(mode="after")
    def _subject_preserved(self) -> QualifiedCapabilityExecutionTarget:
        require_non_empty_text(
            self.qualified_subject_reference,
            label="qualified_subject_reference",
        )
        return self


class QualifiedCapabilityBindingRequest(BaseModel):
    """Binding dispatch envelope — no registry mutation or domain routing in AW."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["qualified_capability_binding_request.v1"] = (
        SCHEMA_QUALIFIED_CAPABILITY_BINDING_REQUEST_V1
    )
    binding_operation_id: str = _NON_EMPTY
    resume_operation_id: str = _NON_EMPTY
    qualified_subject: QualifiedCapabilitySubject
    qualification_result: CapabilityQualificationResult
    worker_need_id: str = _NON_EMPTY
    worker_instance_id: str = _NON_EMPTY
    tenant_id: str = _NON_EMPTY
    task_id: TaskId
    correlation_id: str | None = None
    causation_id: str | None = None
    requested_at: datetime

    @field_validator(
        "binding_operation_id",
        "resume_operation_id",
        "worker_need_id",
        "worker_instance_id",
        "tenant_id",
        "correlation_id",
        "causation_id",
    )
    @classmethod
    def _validate_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_non_empty_text(value, label="text")

    @field_validator("requested_at")
    @classmethod
    def _validate_requested_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("requested_at must be timezone-aware UTC")
        return value

    @model_validator(mode="after")
    def _validate_task(self) -> QualifiedCapabilityBindingRequest:
        validate_task_id(self.task_id)
        return self

    @model_validator(mode="after")
    def _binding_identity(self) -> QualifiedCapabilityBindingRequest:
        expected = derive_qualified_capability_binding_operation_id(
            resume_operation_id=self.resume_operation_id,
            qualified_subject_reference=self.qualified_subject.qualified_subject_reference,
        )
        if self.binding_operation_id != expected:
            raise ValueError("binding_operation_id must match derived identity")
        if (
            self.qualified_subject.qualification_request_id
            != self.qualification_result.qualification_request_id
        ):
            raise ValueError(
                "qualification_result qualification_request_id must match subject",
            )
        return self


class QualifiedCapabilityBindingResult(BaseModel):
    """Immutable auditable binding outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["qualified_capability_binding_result.v1"] = (
        SCHEMA_QUALIFIED_CAPABILITY_BINDING_RESULT_V1
    )
    binding_operation_id: str = _NON_EMPTY
    outcome: QualifiedCapabilityBindingOutcome
    reason_code: QualifiedCapabilityBindingReasonCode
    provider_id: str | None = None
    execution_target: QualifiedCapabilityExecutionTarget | None = None
    started_at: datetime
    completed_at: datetime
    reason_detail: str = ""

    @field_validator("binding_operation_id", "provider_id", "reason_detail")
    @classmethod
    def _validate_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value == "":
            return ""
        return require_non_empty_text(value, label="text")

    @field_validator("started_at", "completed_at")
    @classmethod
    def _validate_timestamps(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware UTC")
        return value

    @model_validator(mode="after")
    def _validate_bound_target(self) -> QualifiedCapabilityBindingResult:
        if self.outcome is QualifiedCapabilityBindingOutcome.BOUND:
            if self.execution_target is None:
                raise ValueError("BOUND requires execution_target")
            if self.provider_id is None:
                raise ValueError("BOUND requires provider_id")
        return self


@runtime_checkable
class QualifiedCapabilityBindingProvider(Protocol):
    """Domain-owned qualified-subject binding SPI."""

    @property
    def provider_id(self) -> str: ...

    def supports(self, request: QualifiedCapabilityBindingRequest) -> bool: ...

    def bind(
        self,
        request: QualifiedCapabilityBindingRequest,
    ) -> QualifiedCapabilityBindingResult: ...


__all__ = [
    "SCHEMA_QUALIFIED_CAPABILITY_BINDING_REQUEST_V1",
    "SCHEMA_QUALIFIED_CAPABILITY_BINDING_RESULT_V1",
    "SCHEMA_QUALIFIED_CAPABILITY_EXECUTION_TARGET_V1",
    "QualifiedCapabilityBindingOutcome",
    "QualifiedCapabilityBindingProvider",
    "QualifiedCapabilityBindingReasonCode",
    "QualifiedCapabilityBindingRequest",
    "QualifiedCapabilityBindingResult",
    "QualifiedCapabilityExecutionTarget",
    "derive_qualified_capability_binding_operation_id",
]
