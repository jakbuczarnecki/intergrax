# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Capability qualification coordination result (UCA-4, AW-7C-P2)."""

from __future__ import annotations

from datetime import datetime
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_qualification.qualification_evidence import (
    CapabilityQualificationEvidence,
)
from intergrax.contracts.capability_qualification.qualification_outcome import (
    CapabilityQualificationOutcome,
)
from intergrax.contracts.capability_qualification.qualification_reason_code import (
    CapabilityQualificationReasonCode,
)
from intergrax.contracts.capability_qualification.qualification_integrity import (
    validate_qualification_evidence_identity,
)
from intergrax.contracts.capability_qualification.qualification_subject import (
    CapabilityQualificationSubjectKind,
)
from intergrax.contracts.capability_qualification.qualification_success_evidence import (
    validate_qualification_success_evidence,
)

SCHEMA_CAPABILITY_QUALIFICATION_RESULT_V1: Final = "capability_qualification_result.v1"
_NON_EMPTY = Field(min_length=1)


class CapabilityQualificationResult(BaseModel):
    """Immutable auditable qualification outcome — not lifecycle execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_qualification_result.v1"] = (
        SCHEMA_CAPABILITY_QUALIFICATION_RESULT_V1
    )
    qualification_request_id: str = _NON_EMPTY
    subject_kind: CapabilityQualificationSubjectKind
    subject_id: str = _NON_EMPTY
    subject_integrity_fingerprint: str = _NON_EMPTY
    tenant_id: str | None = None
    scope_fingerprint: str | None = None
    provider_id: str | None = None
    outcome: CapabilityQualificationOutcome
    reason_code: CapabilityQualificationReasonCode
    started_at: datetime
    completed_at: datetime
    evidence: CapabilityQualificationEvidence | None = None
    reason_detail: str = ""
    correlation_id: str | None = None
    causation_id: str | None = None
    acquisition_request_id: str | None = None
    gap_id: str | None = None
    strategy_id: str | None = None

    @field_validator(
        "qualification_request_id",
        "subject_id",
        "subject_integrity_fingerprint",
        "provider_id",
        "correlation_id",
        "causation_id",
        "reason_detail",
        "acquisition_request_id",
        "gap_id",
        "strategy_id",
        "tenant_id",
        "scope_fingerprint",
    )
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
    def _validate_evidence_chain(self) -> CapabilityQualificationResult:
        if self.evidence is not None:
            if self.provider_id is None:
                raise ValueError("evidence requires provider_id on result")
            validate_qualification_evidence_identity(
                provider_id=self.provider_id,
                qualification_request_id=self.qualification_request_id,
                subject_kind=self.subject_kind,
                subject_id=self.subject_id,
                subject_integrity_fingerprint=self.subject_integrity_fingerprint,
                tenant_id=self.tenant_id,
                scope_fingerprint=self.scope_fingerprint,
                evidence=self.evidence,
            )
            if (
                self.evidence.acquisition_request_id is not None
                and self.acquisition_request_id is not None
                and self.evidence.acquisition_request_id != self.acquisition_request_id
            ):
                raise ValueError(
                    "evidence acquisition_request_id must match result acquisition_request_id",
                )
            if (
                self.evidence.acquisition_strategy_id is not None
                and self.strategy_id is not None
                and self.evidence.acquisition_strategy_id != self.strategy_id
            ):
                raise ValueError(
                    "evidence acquisition_strategy_id must match result strategy_id",
                )
            if (
                self.evidence.gap_id is not None
                and self.gap_id is not None
                and self.evidence.gap_id != self.gap_id
            ):
                raise ValueError("evidence gap_id must match result gap_id")
        if self.outcome is CapabilityQualificationOutcome.QUALIFIED:
            validate_qualification_success_evidence(self.evidence)
            if self.provider_id is None:
                raise ValueError("QUALIFIED requires provider_id")
        if self.subject_kind is CapabilityQualificationSubjectKind.ACQUIRED_CAPABILITY:
            if self.tenant_id is not None or self.scope_fingerprint is not None:
                raise ValueError("ACQUIRED_CAPABILITY result tenant/scope must be absent")
        elif (
            self.subject_kind
            is CapabilityQualificationSubjectKind.SCOPED_INTEGRATION_ADAPTATION
        ):
            if not self.tenant_id or not self.scope_fingerprint:
                raise ValueError(
                    "SCOPED_INTEGRATION_ADAPTATION result requires tenant and scope",
                )
        return self


__all__ = [
    "SCHEMA_CAPABILITY_QUALIFICATION_RESULT_V1",
    "CapabilityQualificationResult",
]
