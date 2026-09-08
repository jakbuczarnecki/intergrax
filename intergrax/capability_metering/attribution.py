# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Discovery-to-usage attribution projection (CAPABILITY-CATALOG-1 Stage 13)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from intergrax.capability_catalog.candidate import CapabilityDiscoveryCandidate
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_catalog.provenance import CapabilityProvenance

SCHEMA_CAPABILITY_USAGE_ATTRIBUTION_V1: Final = "capability_usage_attribution.v1"


class CapabilityUsageAttribution(BaseModel):
    """Immutable attribution snapshot from discovery/selection handoff — not usage itself."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_usage_attribution.v1"] = (
        SCHEMA_CAPABILITY_USAGE_ATTRIBUTION_V1
    )
    identity: CapabilityIdentityKey
    provenance: CapabilityProvenance

    @model_validator(mode="after")
    def _enforce_source_consistency(self) -> CapabilityUsageAttribution:
        if self.identity.source_id != self.provenance.source.source_id:
            raise ValueError(
                "identity.source_id must equal provenance.source.source_id",
            )
        if self.identity.source_kind != self.provenance.source.source_kind:
            raise ValueError(
                "identity.source_kind must equal provenance.source.source_kind",
            )
        return self


def attribution_from_discovery_candidate(
    candidate: CapabilityDiscoveryCandidate,
) -> CapabilityUsageAttribution:
    """Project canonical discovery identity + provenance into usage attribution."""
    return CapabilityUsageAttribution(
        identity=CapabilityIdentityKey.from_discovery_identity(candidate.identity),
        provenance=candidate.provenance,
    )
