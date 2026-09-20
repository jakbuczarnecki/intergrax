# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Immutable discovery completion coordination snapshot (UCA-1).

Projection over canonical Capability Catalog facts — not a second discovery engine
or acquisition orchestrator.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_catalog.federation import (
    CapabilityCatalogFederationCompleteness,
)
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey

SCHEMA_DISCOVERY_COMPLETION_V1: Final = "discovery_completion.v1"
_NON_EMPTY = Field(min_length=1)


class DiscoveryCompletionOutcome(StrEnum):
    """Derived legal branch after completed canonical discovery coordination."""

    DIRECT_REUSE = "direct_reuse"
    REALIZATION_REQUIRED = "realization_required"
    MISSING_CAPABILITY = "missing_capability"
    BLOCKED = "blocked"
    SCOPE_UNAVAILABLE = "scope_unavailable"
    UNAVAILABLE = "unavailable"
    CONFLICT = "conflict"
    INCOMPLETE = "incomplete"


NORMATIVE_DISCOVERY_COMPLETION_OUTCOMES: Final[frozenset[DiscoveryCompletionOutcome]] = (
    frozenset(DiscoveryCompletionOutcome)
)


def derive_discovery_completion_outcome(
    *,
    federation_completeness: CapabilityCatalogFederationCompleteness,
    suitable_host_allowed_keys: tuple[CapabilityIdentityKey, ...],
    suitable_catalog_allowed_keys: tuple[CapabilityIdentityKey, ...],
    governance_blocked: bool,
    availability_blocked: bool,
    scope_unavailable: bool,
    unavailable: bool,
    conflict: bool,
) -> DiscoveryCompletionOutcome:
    """Derive the coordination branch from separated source facts.

    Suitability, availability, governance, and federation completeness remain
    independent inputs — this function only projects the legal next branch.
    """
    if conflict:
        return DiscoveryCompletionOutcome.CONFLICT
    if suitable_host_allowed_keys:
        return DiscoveryCompletionOutcome.DIRECT_REUSE
    if suitable_catalog_allowed_keys:
        return DiscoveryCompletionOutcome.REALIZATION_REQUIRED
    if (
        federation_completeness
        is CapabilityCatalogFederationCompleteness.PARTIAL
    ):
        return DiscoveryCompletionOutcome.INCOMPLETE
    if scope_unavailable:
        return DiscoveryCompletionOutcome.SCOPE_UNAVAILABLE
    if unavailable:
        return DiscoveryCompletionOutcome.UNAVAILABLE
    if governance_blocked or availability_blocked:
        return DiscoveryCompletionOutcome.BLOCKED
    return DiscoveryCompletionOutcome.MISSING_CAPABILITY


def _dedupe_identity_keys(
    keys: tuple[CapabilityIdentityKey, ...],
    *,
    label: str,
) -> tuple[CapabilityIdentityKey, ...]:
    seen: set[tuple[str, str, str, str]] = set()
    ordered: list[CapabilityIdentityKey] = []
    for key in keys:
        sort_key = key.sort_key
        if sort_key in seen:
            raise ValueError(f"{label} must not repeat the same identity key")
        seen.add(sort_key)
        ordered.append(key)
    return tuple(sorted(ordered, key=lambda item: item.sort_key))


class DiscoveryCompletion(BaseModel):
    """Evidence-backed coordination snapshot — not a discovery source of truth."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["discovery_completion.v1"] = SCHEMA_DISCOVERY_COMPLETION_V1
    need_id: str = _NON_EMPTY
    discovery_correlation_id: str = _NON_EMPTY
    federation_completeness: CapabilityCatalogFederationCompleteness
    suitable_host_allowed_keys: tuple[CapabilityIdentityKey, ...] = ()
    suitable_catalog_allowed_keys: tuple[CapabilityIdentityKey, ...] = ()
    governance_blocked: bool = False
    availability_blocked: bool = False
    scope_unavailable: bool = False
    unavailable: bool = False
    conflict: bool = False
    outcome: DiscoveryCompletionOutcome
    created_at: datetime

    @field_validator("need_id", "discovery_correlation_id")
    @classmethod
    def _validate_ids(cls, value: str) -> str:
        return require_non_empty_text(value, label="id")

    @field_validator("suitable_host_allowed_keys", "suitable_catalog_allowed_keys")
    @classmethod
    def _validate_keys(
        cls,
        value: tuple[CapabilityIdentityKey, ...],
    ) -> tuple[CapabilityIdentityKey, ...]:
        return _dedupe_identity_keys(value, label="identity keys")

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware UTC")
        return value

    @model_validator(mode="after")
    def _validate_outcome_and_disjoint_keys(self) -> DiscoveryCompletion:
        host_keys = {key.sort_key for key in self.suitable_host_allowed_keys}
        catalog_keys = {key.sort_key for key in self.suitable_catalog_allowed_keys}
        overlap = host_keys & catalog_keys
        if overlap:
            raise ValueError(
                "suitable_host_allowed_keys and suitable_catalog_allowed_keys "
                f"must be disjoint; overlap={min(overlap)!r}",
            )
        expected = derive_discovery_completion_outcome(
            federation_completeness=self.federation_completeness,
            suitable_host_allowed_keys=self.suitable_host_allowed_keys,
            suitable_catalog_allowed_keys=self.suitable_catalog_allowed_keys,
            governance_blocked=self.governance_blocked,
            availability_blocked=self.availability_blocked,
            scope_unavailable=self.scope_unavailable,
            unavailable=self.unavailable,
            conflict=self.conflict,
        )
        if self.outcome is not expected:
            raise ValueError(
                f"outcome {self.outcome.value!r} is inconsistent with source facts; "
                f"expected {expected.value!r}",
            )
        return self


def build_discovery_completion(
    *,
    need_id: str,
    discovery_correlation_id: str,
    federation_completeness: CapabilityCatalogFederationCompleteness,
    created_at: datetime,
    suitable_host_allowed_keys: tuple[CapabilityIdentityKey, ...] = (),
    suitable_catalog_allowed_keys: tuple[CapabilityIdentityKey, ...] = (),
    governance_blocked: bool = False,
    availability_blocked: bool = False,
    scope_unavailable: bool = False,
    unavailable: bool = False,
    conflict: bool = False,
) -> DiscoveryCompletion:
    """Build a completion snapshot with derived outcome from source facts."""
    outcome = derive_discovery_completion_outcome(
        federation_completeness=federation_completeness,
        suitable_host_allowed_keys=suitable_host_allowed_keys,
        suitable_catalog_allowed_keys=suitable_catalog_allowed_keys,
        governance_blocked=governance_blocked,
        availability_blocked=availability_blocked,
        scope_unavailable=scope_unavailable,
        unavailable=unavailable,
        conflict=conflict,
    )
    return DiscoveryCompletion(
        need_id=need_id,
        discovery_correlation_id=discovery_correlation_id,
        federation_completeness=federation_completeness,
        suitable_host_allowed_keys=suitable_host_allowed_keys,
        suitable_catalog_allowed_keys=suitable_catalog_allowed_keys,
        governance_blocked=governance_blocked,
        availability_blocked=availability_blocked,
        scope_unavailable=scope_unavailable,
        unavailable=unavailable,
        conflict=conflict,
        outcome=outcome,
        created_at=created_at,
    )


__all__ = [
    "SCHEMA_DISCOVERY_COMPLETION_V1",
    "DiscoveryCompletion",
    "DiscoveryCompletionOutcome",
    "NORMATIVE_DISCOVERY_COMPLETION_OUTCOMES",
    "build_discovery_completion",
    "derive_discovery_completion_outcome",
]
