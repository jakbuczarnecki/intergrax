# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Effective work-stage capability set and discovery evidence (Stage 8)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from intergrax.capability_catalog.governed_candidate import GovernedCapabilityCandidate
from intergrax.capability_catalog.governed_result import GovernedDiscoveryResult
from intergrax.contracts.capability_catalog.availability import AvailabilityDisposition
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_catalog.work_stage import WorkStageCapabilityNeed

SCHEMA_EFFECTIVE_CAPABILITY_SET_V1: Final = "effective_capability_set.v1"
SCHEMA_WORK_STAGE_CAPABILITY_DISCOVERY_EVIDENCE_V1: Final = (
    "work_stage_capability_discovery_evidence.v1"
)
SCHEMA_WORK_STAGE_CAPABILITY_TRANSITION_EVIDENCE_V1: Final = (
    "work_stage_capability_transition_evidence.v1"
)


class EffectiveCapabilitySet(BaseModel):
    """Deterministic query result — not runtime inventory authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["effective_capability_set.v1"] = SCHEMA_EFFECTIVE_CAPABILITY_SET_V1
    need: WorkStageCapabilityNeed
    governed_result: GovernedDiscoveryResult
    effective_candidates: tuple[GovernedCapabilityCandidate, ...]

    @model_validator(mode="after")
    def _validate_effective_candidates(self) -> EffectiveCapabilitySet:
        for candidate in self.effective_candidates:
            if candidate.availability is not AvailabilityDisposition.HOST_AVAILABLE:
                raise ValueError(
                    "effective candidates must be HOST_AVAILABLE executable members",
                )
        return self

    @property
    def effective_identity_keys(self) -> tuple[CapabilityIdentityKey, ...]:
        return tuple(
            CapabilityIdentityKey.from_discovery_identity(candidate.identity)
            for candidate in self.effective_candidates
        )


class WorkStageCapabilityDiscoveryEvidence(BaseModel):
    """Deterministic rediscovery evidence for one work stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["work_stage_capability_discovery_evidence.v1"] = (
        SCHEMA_WORK_STAGE_CAPABILITY_DISCOVERY_EVIDENCE_V1
    )
    need: WorkStageCapabilityNeed
    effective_set: EffectiveCapabilitySet
    catalog_only_identity_keys: tuple[CapabilityIdentityKey, ...] = ()

    @model_validator(mode="after")
    def _validate_need_alignment(self) -> WorkStageCapabilityDiscoveryEvidence:
        if self.need != self.effective_set.need:
            raise ValueError("evidence need must match effective set need")
        return self


class WorkStageCapabilityTransitionEvidence(BaseModel):
    """Typed comparison between consecutive stage effective sets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["work_stage_capability_transition_evidence.v1"] = (
        SCHEMA_WORK_STAGE_CAPABILITY_TRANSITION_EVIDENCE_V1
    )
    previous: EffectiveCapabilitySet
    current: EffectiveCapabilitySet
    added_identity_keys: tuple[CapabilityIdentityKey, ...]
    removed_identity_keys: tuple[CapabilityIdentityKey, ...]

    @model_validator(mode="after")
    def _validate_work_alignment(self) -> WorkStageCapabilityTransitionEvidence:
        if self.previous.need.work_reference != self.current.need.work_reference:
            raise ValueError("transition evidence requires the same work_reference")
        return self


def compare_work_stage_effective_capabilities(
    previous: EffectiveCapabilitySet,
    current: EffectiveCapabilitySet,
) -> WorkStageCapabilityTransitionEvidence:
    """Derive added/removed identity keys between two stage results."""
    previous_keys = frozenset(key.sort_key for key in previous.effective_identity_keys)
    current_keys = frozenset(key.sort_key for key in current.effective_identity_keys)
    added = current_keys - previous_keys
    removed = previous_keys - current_keys
    added_keys = tuple(
        key
        for key in current.effective_identity_keys
        if key.sort_key in added
    )
    removed_keys = tuple(
        key
        for key in previous.effective_identity_keys
        if key.sort_key in removed
    )
    return WorkStageCapabilityTransitionEvidence(
        previous=previous,
        current=current,
        added_identity_keys=added_keys,
        removed_identity_keys=removed_keys,
    )
