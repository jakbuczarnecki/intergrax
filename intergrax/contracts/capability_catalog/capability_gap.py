# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Capability gap contract — semantic absence after complete discovery (UCA-1)."""

from __future__ import annotations

from datetime import datetime
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_catalog.discovery_completion import (
    DiscoveryCompletion,
    DiscoveryCompletionOutcome,
)
from intergrax.contracts.capability_catalog.federation import (
    CapabilityCatalogFederationCompleteness,
)

SCHEMA_CAPABILITY_GAP_V1: Final = "capability_gap.v1"
_NON_EMPTY = Field(min_length=1)


def derive_capability_gap_id(
    *,
    need_id: str,
    discovery_correlation_id: str,
) -> str:
    """Deterministic gap identity from canonical coordination inputs."""
    normalized_need = require_non_empty_text(need_id, label="need_id")
    normalized_correlation = require_non_empty_text(
        discovery_correlation_id,
        label="discovery_correlation_id",
    )
    return f"capability-gap:{normalized_need}:{normalized_correlation}"


class CapabilityGap(BaseModel):
    """Semantic absence after sufficiently complete canonical discovery.

    Not a catch-all for blocked, unavailable, conflict, or partial federation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_gap.v1"] = SCHEMA_CAPABILITY_GAP_V1
    gap_id: str = _NON_EMPTY
    need_id: str = _NON_EMPTY
    discovery_correlation_id: str = _NON_EMPTY
    federation_completeness: CapabilityCatalogFederationCompleteness = (
        CapabilityCatalogFederationCompleteness.COMPLETE
    )
    created_at: datetime

    @field_validator("gap_id", "need_id", "discovery_correlation_id")
    @classmethod
    def _validate_ids(cls, value: str) -> str:
        return require_non_empty_text(value, label="id")

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware UTC")
        return value

    @model_validator(mode="after")
    def _validate_gap_identity(self) -> CapabilityGap:
        if (
            self.federation_completeness
            is not CapabilityCatalogFederationCompleteness.COMPLETE
        ):
            raise ValueError(
                "CapabilityGap requires COMPLETE federation completeness",
            )
        expected = derive_capability_gap_id(
            need_id=self.need_id,
            discovery_correlation_id=self.discovery_correlation_id,
        )
        if self.gap_id != expected:
            raise ValueError(
                f"gap_id must be deterministic derived identity {expected!r}",
            )
        return self

    @classmethod
    def from_discovery_completion(cls, completion: DiscoveryCompletion) -> CapabilityGap:
        """Fail-closed construction — Gap only from MISSING_CAPABILITY completion."""
        if completion.outcome is not DiscoveryCompletionOutcome.MISSING_CAPABILITY:
            raise ValueError(
                "CapabilityGap requires DiscoveryCompletionOutcome.MISSING_CAPABILITY; "
                f"got {completion.outcome.value}",
            )
        if (
            completion.federation_completeness
            is not CapabilityCatalogFederationCompleteness.COMPLETE
        ):
            raise ValueError(
                "CapabilityGap requires COMPLETE federation; "
                f"got {completion.federation_completeness.value}",
            )
        if completion.suitable_host_allowed_keys:
            raise ValueError(
                "CapabilityGap forbids HOST_AVAILABLE suitable candidates",
            )
        if completion.suitable_catalog_allowed_keys:
            raise ValueError(
                "CapabilityGap forbids CATALOG_AVAILABLE suitable candidates",
            )
        if completion.conflict:
            raise ValueError("CapabilityGap forbids CONFLICT discovery")
        if completion.scope_unavailable:
            raise ValueError("CapabilityGap forbids SCOPE_UNAVAILABLE")
        if completion.unavailable:
            raise ValueError("CapabilityGap forbids UNAVAILABLE")
        if completion.governance_blocked or completion.availability_blocked:
            raise ValueError("CapabilityGap forbids BLOCKED discovery")
        return cls(
            gap_id=derive_capability_gap_id(
                need_id=completion.need_id,
                discovery_correlation_id=completion.discovery_correlation_id,
            ),
            need_id=completion.need_id,
            discovery_correlation_id=completion.discovery_correlation_id,
            created_at=completion.created_at,
        )


__all__ = [
    "SCHEMA_CAPABILITY_GAP_V1",
    "CapabilityGap",
    "derive_capability_gap_id",
]
