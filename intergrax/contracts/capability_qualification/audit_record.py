# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed audit chain linking acquisition/adaptation, qualification, and lifecycle."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_qualification.lifecycle_decision import (
    CapabilityQualificationLifecycleDecision,
    CapabilityQualificationLifecycleOutcome,
)
from intergrax.contracts.capability_qualification.qualification_outcome import (
    CapabilityQualificationOutcome,
)
from intergrax.contracts.capability_qualification.qualification_subject import (
    CapabilityQualificationSubjectKind,
)

SCHEMA_CAPABILITY_QUALIFICATION_AUDIT_RECORD_V1: Final = (
    "capability_qualification_audit_record.v1"
)
_NON_EMPTY = Field(min_length=1)


class CapabilityQualificationAuditRecord(BaseModel):
    """Semantic provenance anchor — not application logging."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_qualification_audit_record.v1"] = (
        SCHEMA_CAPABILITY_QUALIFICATION_AUDIT_RECORD_V1
    )
    qualification_request_id: str = _NON_EMPTY
    subject_kind: CapabilityQualificationSubjectKind
    subject_id: str = _NON_EMPTY
    subject_integrity_fingerprint: str = _NON_EMPTY
    tenant_id: str | None = None
    scope_fingerprint: str | None = None
    acquisition_request_id: str | None = None
    acquisition_strategy_id: str | None = None
    gap_id: str | None = None
    qualification_provider_id: str | None = None
    qualification_outcome: CapabilityQualificationOutcome
    lifecycle_outcome: CapabilityQualificationLifecycleOutcome
    correlation_id: str | None = None
    causation_id: str | None = None

    @field_validator(
        "qualification_request_id",
        "subject_id",
        "subject_integrity_fingerprint",
        "acquisition_request_id",
        "acquisition_strategy_id",
        "gap_id",
        "qualification_provider_id",
        "correlation_id",
        "causation_id",
        "tenant_id",
        "scope_fingerprint",
    )
    @classmethod
    def _validate_ids(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_non_empty_text(value, label="id")


def build_qualification_audit_record(
    *,
    qualification_request_id: str,
    subject_kind: CapabilityQualificationSubjectKind,
    subject_id: str,
    subject_integrity_fingerprint: str,
    tenant_id: str | None,
    scope_fingerprint: str | None,
    acquisition_request_id: str | None,
    acquisition_strategy_id: str | None,
    gap_id: str | None,
    qualification_provider_id: str | None,
    qualification_outcome: CapabilityQualificationOutcome,
    lifecycle_decision: CapabilityQualificationLifecycleDecision,
    correlation_id: str | None,
    causation_id: str | None,
) -> CapabilityQualificationAuditRecord:
    return CapabilityQualificationAuditRecord(
        qualification_request_id=qualification_request_id,
        subject_kind=subject_kind,
        subject_id=subject_id,
        subject_integrity_fingerprint=subject_integrity_fingerprint,
        tenant_id=tenant_id,
        scope_fingerprint=scope_fingerprint,
        acquisition_request_id=acquisition_request_id,
        acquisition_strategy_id=acquisition_strategy_id,
        gap_id=gap_id,
        qualification_provider_id=qualification_provider_id,
        qualification_outcome=qualification_outcome,
        lifecycle_outcome=lifecycle_decision.outcome,
        correlation_id=correlation_id,
        causation_id=causation_id,
    )


__all__ = [
    "SCHEMA_CAPABILITY_QUALIFICATION_AUDIT_RECORD_V1",
    "CapabilityQualificationAuditRecord",
    "build_qualification_audit_record",
]
