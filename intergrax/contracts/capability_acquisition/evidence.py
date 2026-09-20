# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Realization evidence — availability facts for canonical reconfirmation (UCA-2)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.capability_catalog.evidence import (
    CapabilityDiscoveryAvailabilityEvidence,
)

SCHEMA_CAPABILITY_REALIZATION_EVIDENCE_V1: Final = "capability_realization_evidence.v1"
_NON_EMPTY = Field(min_length=1)


class CapabilityRealizationEvidence(BaseModel):
    """Domain-supplied facts enabling canonical HOST_AVAILABLE projection.

    UCA MUST NOT treat provider success without this evidence as HOST_AVAILABLE.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_realization_evidence.v1"] = (
        SCHEMA_CAPABILITY_REALIZATION_EVIDENCE_V1
    )
    domain_reference: str | None = None
    availability_evidence: CapabilityDiscoveryAvailabilityEvidence | None = None
    evidence_ref: str | None = None

    @classmethod
    def from_availability_evidence(
        cls,
        evidence: CapabilityDiscoveryAvailabilityEvidence,
        *,
        domain_reference: str | None = None,
        evidence_ref: str | None = None,
    ) -> CapabilityRealizationEvidence:
        return cls(
            domain_reference=domain_reference,
            availability_evidence=evidence,
            evidence_ref=evidence_ref,
        )


__all__ = [
    "SCHEMA_CAPABILITY_REALIZATION_EVIDENCE_V1",
    "CapabilityRealizationEvidence",
]
