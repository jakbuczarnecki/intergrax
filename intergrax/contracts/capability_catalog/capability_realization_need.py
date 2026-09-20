# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Capability realization need — known capability not yet HOST_AVAILABLE (UCA-1)."""

from __future__ import annotations

from datetime import datetime
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_catalog.availability import AvailabilityDisposition
from intergrax.contracts.capability_catalog.discovery_completion import (
    DiscoveryCompletion,
    DiscoveryCompletionOutcome,
)
from intergrax.contracts.capability_catalog.governance import GovernanceDisposition
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey

SCHEMA_CAPABILITY_REALIZATION_NEED_V1: Final = "capability_realization_need.v1"
_NON_EMPTY = Field(min_length=1)


def derive_capability_realization_need_id(
    *,
    need_id: str,
    discovery_correlation_id: str,
    capability_identity: CapabilityIdentityKey,
) -> str:
    """Deterministic realization-need identity from need + known capability key."""
    normalized_need = require_non_empty_text(need_id, label="need_id")
    normalized_correlation = require_non_empty_text(
        discovery_correlation_id,
        label="discovery_correlation_id",
    )
    kind, source_id, source_kind, logical_id = capability_identity.sort_key
    return (
        "capability-realization:"
        f"{normalized_need}:"
        f"{normalized_correlation}:"
        f"{kind}:{source_id}:{source_kind}:{logical_id}"
    )


class CapabilityRealizationNeed(BaseModel):
    """Need to bring a known, suitable, governance-allowed capability to usable state.

    Does not guarantee installation, activation, authorization, or execution.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_realization_need.v1"] = (
        SCHEMA_CAPABILITY_REALIZATION_NEED_V1
    )
    realization_need_id: str = _NON_EMPTY
    need_id: str = _NON_EMPTY
    discovery_correlation_id: str = _NON_EMPTY
    capability_identity: CapabilityIdentityKey
    availability: Literal[AvailabilityDisposition.CATALOG_AVAILABLE] = (
        AvailabilityDisposition.CATALOG_AVAILABLE
    )
    governance_disposition: Literal[GovernanceDisposition.ALLOWED] = (
        GovernanceDisposition.ALLOWED
    )
    created_at: datetime

    @field_validator("realization_need_id", "need_id", "discovery_correlation_id")
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
    def _validate_realization_identity(self) -> CapabilityRealizationNeed:
        expected = derive_capability_realization_need_id(
            need_id=self.need_id,
            discovery_correlation_id=self.discovery_correlation_id,
            capability_identity=self.capability_identity,
        )
        if self.realization_need_id != expected:
            raise ValueError(
                "realization_need_id must be deterministic derived identity "
                f"{expected!r}",
            )
        return self

    @classmethod
    def from_discovery_completion(
        cls,
        completion: DiscoveryCompletion,
        *,
        capability_identity: CapabilityIdentityKey | None = None,
    ) -> CapabilityRealizationNeed:
        """Fail-closed construction from REALIZATION_REQUIRED completion."""
        if completion.outcome is not DiscoveryCompletionOutcome.REALIZATION_REQUIRED:
            raise ValueError(
                "CapabilityRealizationNeed requires "
                "DiscoveryCompletionOutcome.REALIZATION_REQUIRED; "
                f"got {completion.outcome.value}",
            )
        catalog_keys = completion.suitable_catalog_allowed_keys
        if not catalog_keys:
            raise ValueError(
                "CapabilityRealizationNeed requires at least one "
                "CATALOG_AVAILABLE allowed suitable identity",
            )
        selected = capability_identity or catalog_keys[0]
        catalog_sort_keys = {key.sort_key for key in catalog_keys}
        if selected.sort_key not in catalog_sort_keys:
            raise ValueError(
                "capability_identity must be one of suitable_catalog_allowed_keys",
            )
        host_sort_keys = {
            key.sort_key for key in completion.suitable_host_allowed_keys
        }
        if selected.sort_key in host_sort_keys:
            raise ValueError(
                "CapabilityRealizationNeed forbids HOST_AVAILABLE identity",
            )
        return cls(
            realization_need_id=derive_capability_realization_need_id(
                need_id=completion.need_id,
                discovery_correlation_id=completion.discovery_correlation_id,
                capability_identity=selected,
            ),
            need_id=completion.need_id,
            discovery_correlation_id=completion.discovery_correlation_id,
            capability_identity=selected,
            created_at=completion.created_at,
        )


__all__ = [
    "SCHEMA_CAPABILITY_REALIZATION_NEED_V1",
    "CapabilityRealizationNeed",
    "derive_capability_realization_need_id",
]
