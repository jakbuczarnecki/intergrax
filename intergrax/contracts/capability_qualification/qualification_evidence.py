# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Qualification and provenance evidence — not logging (UCA-4, AW-7C-P2)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intergrax.contracts.capability_qualification.qualification_subject import (
    CapabilityQualificationSubjectKind,
)

SCHEMA_CAPABILITY_QUALIFICATION_EVIDENCE_V1: Final = (
    "capability_qualification_evidence.v1"
)
_NON_EMPTY = Field(min_length=1)


class CapabilityQualificationEvidence(BaseModel):
    """Typed qualification facts — provider-produced, policy-evaluated separately."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_qualification_evidence.v1"] = (
        SCHEMA_CAPABILITY_QUALIFICATION_EVIDENCE_V1
    )
    provider_id: str = _NON_EMPTY
    qualification_request_id: str = _NON_EMPTY
    subject_kind: CapabilityQualificationSubjectKind
    subject_id: str = _NON_EMPTY
    subject_integrity_fingerprint: str = _NON_EMPTY
    tenant_id: str | None = None
    scope_fingerprint: str | None = None
    artifact_reference: str | None = None
    domain_handoff_reference: str | None = None
    evidence_ref: str | None = None
    verification_refs: tuple[str, ...] = ()
    acquisition_request_id: str | None = None
    acquisition_strategy_id: str | None = None
    gap_id: str | None = None

    @model_validator(mode="after")
    def _tenant_scope_consistent(self) -> CapabilityQualificationEvidence:
        if self.subject_kind is CapabilityQualificationSubjectKind.ACQUIRED_CAPABILITY:
            if self.tenant_id is not None:
                raise ValueError("ACQUIRED_CAPABILITY evidence must not carry tenant_id")
            if self.scope_fingerprint is not None:
                raise ValueError(
                    "ACQUIRED_CAPABILITY evidence must not carry scope_fingerprint",
                )
            return self
        if (
            self.subject_kind
            is CapabilityQualificationSubjectKind.SCOPED_INTEGRATION_ADAPTATION
        ):
            if not self.tenant_id:
                raise ValueError(
                    "SCOPED_INTEGRATION_ADAPTATION evidence requires tenant_id",
                )
            if not self.scope_fingerprint:
                raise ValueError(
                    "SCOPED_INTEGRATION_ADAPTATION evidence requires scope_fingerprint",
                )
            return self
        raise ValueError(f"unsupported subject_kind: {self.subject_kind}")


__all__ = [
    "SCHEMA_CAPABILITY_QUALIFICATION_EVIDENCE_V1",
    "CapabilityQualificationEvidence",
]
