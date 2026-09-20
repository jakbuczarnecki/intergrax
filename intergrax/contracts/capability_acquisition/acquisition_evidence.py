# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Acquisition handoff evidence — not HOST_AVAILABLE or execution proof (UCA-3)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_CAPABILITY_ACQUISITION_EVIDENCE_V1: Final = "capability_acquisition_evidence.v1"
_NON_EMPTY = Field(min_length=1)


class CapabilityAcquisitionEvidence(BaseModel):
    """Opaque domain handoff or artifact reference after strategy acquisition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_acquisition_evidence.v1"] = (
        SCHEMA_CAPABILITY_ACQUISITION_EVIDENCE_V1
    )
    domain_handoff_reference: str | None = None
    artifact_reference: str | None = None
    evidence_ref: str | None = None


__all__ = [
    "SCHEMA_CAPABILITY_ACQUISITION_EVIDENCE_V1",
    "CapabilityAcquisitionEvidence",
]
