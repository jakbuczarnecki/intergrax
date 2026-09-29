# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Capability qualification coordination request (UCA-4, AW-7C-P2 subject-oriented)."""

from __future__ import annotations

from datetime import datetime
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.capability_acquisition.acquisition_result import (
    CapabilityAcquisitionResult,
)
from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_qualification.qualification_subject import (
    CapabilityQualificationSubject,
    project_acquisition_qualification_subject,
)

SCHEMA_CAPABILITY_QUALIFICATION_REQUEST_V1: Final = (
    "capability_qualification_request.v1"
)
_NON_EMPTY = Field(min_length=1)


def derive_capability_qualification_request_id(
    *,
    subject_id: str,
    qualification_nonce: str,
) -> str:
    """Deterministic qualification identity from canonical subject id."""
    normalized_subject = require_non_empty_text(subject_id, label="subject_id")
    normalized_nonce = require_non_empty_text(
        qualification_nonce,
        label="qualification_nonce",
    )
    return (
        f"capability-qualification-request:{normalized_subject}:{normalized_nonce}"
    )


def build_subject_qualification_request(
    *,
    subject: CapabilityQualificationSubject,
    qualification_nonce: str,
    requested_at: datetime,
) -> CapabilityQualificationRequest:
    lineage = subject.acquisition_lineage
    adaptation = subject.adaptation_lineage
    if lineage is not None:
        correlation_id = lineage.correlation_id
        causation_id = lineage.causation_id
    elif adaptation is not None:
        correlation_id = adaptation.correlation_id
        causation_id = adaptation.causation_id
    else:
        raise ValueError("subject must carry acquisition or adaptation lineage")
    return CapabilityQualificationRequest(
        qualification_request_id=derive_capability_qualification_request_id(
            subject_id=subject.subject_id,
            qualification_nonce=qualification_nonce,
        ),
        qualification_nonce=qualification_nonce,
        subject=subject,
        correlation_id=correlation_id,
        causation_id=causation_id,
        requested_at=requested_at,
    )


def build_acquisition_qualification_request(
    *,
    acquisition_result: CapabilityAcquisitionResult,
    qualification_nonce: str,
    requested_at: datetime,
) -> CapabilityQualificationRequest:
    subject = project_acquisition_qualification_subject(
        acquisition_result=acquisition_result,
    )
    lineage = subject.acquisition_lineage
    if lineage is None:
        raise ValueError("acquisition subject missing lineage")
    return CapabilityQualificationRequest(
        qualification_request_id=derive_capability_qualification_request_id(
            subject_id=subject.subject_id,
            qualification_nonce=qualification_nonce,
        ),
        qualification_nonce=qualification_nonce,
        subject=subject,
        correlation_id=lineage.correlation_id,
        causation_id=lineage.causation_id,
        requested_at=requested_at,
    )


class CapabilityQualificationRequest(BaseModel):
    """Qualification dispatch envelope — canonical subject is semantic authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_qualification_request.v1"] = (
        SCHEMA_CAPABILITY_QUALIFICATION_REQUEST_V1
    )
    qualification_request_id: str = _NON_EMPTY
    qualification_nonce: str = _NON_EMPTY
    subject: CapabilityQualificationSubject
    correlation_id: str | None = None
    causation_id: str | None = None
    requested_at: datetime

    @field_validator(
        "qualification_request_id",
        "qualification_nonce",
        "correlation_id",
        "causation_id",
    )
    @classmethod
    def _validate_ids(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_non_empty_text(value, label="id")

    @field_validator("requested_at")
    @classmethod
    def _validate_requested_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("requested_at must be timezone-aware UTC")
        return value

    @model_validator(mode="after")
    def _validate_handoff_boundary(self) -> CapabilityQualificationRequest:
        expected_id = derive_capability_qualification_request_id(
            subject_id=self.subject.subject_id,
            qualification_nonce=self.qualification_nonce,
        )
        if self.qualification_request_id != expected_id:
            raise ValueError(
                f"qualification_request_id must be derived identity {expected_id!r}",
            )
        lineage = self.subject.acquisition_lineage
        if lineage is not None:
            if self.correlation_id != lineage.correlation_id:
                raise ValueError("correlation_id must match acquisition lineage")
            if self.causation_id != lineage.causation_id:
                raise ValueError("causation_id must match acquisition lineage")
        adaptation = self.subject.adaptation_lineage
        if adaptation is not None:
            if self.correlation_id != adaptation.correlation_id:
                raise ValueError("correlation_id must match adaptation lineage")
            if self.causation_id != adaptation.causation_id:
                raise ValueError("causation_id must match adaptation lineage")
        return self


__all__ = [
    "SCHEMA_CAPABILITY_QUALIFICATION_REQUEST_V1",
    "CapabilityQualificationRequest",
    "build_acquisition_qualification_request",
    "build_subject_qualification_request",
    "derive_capability_qualification_request_id",
]
