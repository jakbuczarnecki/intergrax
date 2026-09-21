# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Qualified capability subject identity — opaque to Autonomous Work (UCA-6C)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_qualification.qualification_outcome import (
    CapabilityQualificationOutcome,
)
from intergrax.contracts.capability_qualification.qualification_result import (
    CapabilityQualificationResult,
)

SCHEMA_QUALIFIED_CAPABILITY_SUBJECT_V1: Final = "qualified_capability_subject.v1"
_NON_EMPTY = Field(min_length=1)


class QualifiedCapabilitySubjectKind(StrEnum):
    """Which evidence field names the qualified subject — not a provider or domain."""

    ARTIFACT_REFERENCE = "artifact_reference"
    DOMAIN_HANDOFF_REFERENCE = "domain_handoff_reference"
    EVIDENCE_REF = "evidence_ref"


class QualifiedCapabilitySubject(BaseModel):
    """Immutable qualified subject — AW must not interpret ``subject_reference``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["qualified_capability_subject.v1"] = (
        SCHEMA_QUALIFIED_CAPABILITY_SUBJECT_V1
    )
    qualification_request_id: str = _NON_EMPTY
    subject_kind: QualifiedCapabilitySubjectKind
    subject_reference: str = _NON_EMPTY
    qualified_subject_reference: str = _NON_EMPTY

    @model_validator(mode="after")
    def _identity_consistent(self) -> QualifiedCapabilitySubject:
        expected = derive_qualified_subject_reference(
            qualification_request_id=self.qualification_request_id,
            subject_kind=self.subject_kind,
            subject_reference=self.subject_reference,
        )
        if self.qualified_subject_reference != expected:
            raise ValueError("qualified_subject_reference must match derived identity")
        return self


def derive_qualified_subject_reference(
    *,
    qualification_request_id: str,
    subject_kind: QualifiedCapabilitySubjectKind,
    subject_reference: str,
) -> str:
    qid = require_non_empty_text(
        qualification_request_id,
        label="qualification_request_id",
    )
    ref = require_non_empty_text(subject_reference, label="subject_reference")
    return f"qualified-capability-subject:{qid}:{subject_kind.value}:{ref}"


def qualified_capability_subject_from_result(
    result: CapabilityQualificationResult,
) -> QualifiedCapabilitySubject | None:
    """Extract subject only from QUALIFIED results with typed evidence."""
    if result.outcome is not CapabilityQualificationOutcome.QUALIFIED:
        return None
    evidence = result.evidence
    if evidence is None:
        return None
    if evidence.artifact_reference:
        kind = QualifiedCapabilitySubjectKind.ARTIFACT_REFERENCE
        ref = evidence.artifact_reference
    elif evidence.domain_handoff_reference:
        kind = QualifiedCapabilitySubjectKind.DOMAIN_HANDOFF_REFERENCE
        ref = evidence.domain_handoff_reference
    elif evidence.evidence_ref:
        kind = QualifiedCapabilitySubjectKind.EVIDENCE_REF
        ref = evidence.evidence_ref
    else:
        return None
    return QualifiedCapabilitySubject(
        qualification_request_id=result.qualification_request_id,
        subject_kind=kind,
        subject_reference=ref,
        qualified_subject_reference=derive_qualified_subject_reference(
            qualification_request_id=result.qualification_request_id,
            subject_kind=kind,
            subject_reference=ref,
        ),
    )


__all__ = [
    "SCHEMA_QUALIFIED_CAPABILITY_SUBJECT_V1",
    "QualifiedCapabilitySubject",
    "QualifiedCapabilitySubjectKind",
    "derive_qualified_subject_reference",
    "qualified_capability_subject_from_result",
]
