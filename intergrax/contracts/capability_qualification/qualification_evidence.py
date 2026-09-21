# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Qualification and provenance evidence — not logging (UCA-4)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

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
    acquisition_request_id: str = _NON_EMPTY
    acquisition_strategy_id: str = _NON_EMPTY
    gap_id: str = _NON_EMPTY
    artifact_reference: str | None = None
    domain_handoff_reference: str | None = None
    evidence_ref: str | None = None
    verification_refs: tuple[str, ...] = ()


__all__ = [
    "SCHEMA_CAPABILITY_QUALIFICATION_EVIDENCE_V1",
    "CapabilityQualificationEvidence",
]
