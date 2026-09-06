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


def _catalog_only_identity_keys(
    allowed: tuple[GovernedCapabilityCandidate, ...],
    effective: tuple[GovernedCapabilityCandidate, ...],
) -> tuple[CapabilityIdentityKey, ...]:
    effective_keys = frozenset(
        candidate.ranked.identity.sort_key for candidate in effective
    )
    catalog_only = tuple(
        CapabilityIdentityKey.from_discovery_identity(candidate.identity)
        for candidate in allowed
        if candidate.availability is AvailabilityDisposition.CATALOG_AVAILABLE
        and candidate.ranked.identity.sort_key not in effective_keys
    )
    return tuple(sorted(catalog_only, key=lambda key: key.sort_key))


def _transition_identity_diff(
    previous_keys: tuple[CapabilityIdentityKey, ...],
    current_keys: tuple[CapabilityIdentityKey, ...],
) -> tuple[tuple[CapabilityIdentityKey, ...], tuple[CapabilityIdentityKey, ...]]:
    previous_set = frozenset(key.sort_key for key in previous_keys)
    current_set = frozenset(key.sort_key for key in current_keys)
    added = current_set - previous_set
    removed = previous_set - current_set
    added_keys = tuple(key for key in current_keys if key.sort_key in added)
    removed_keys = tuple(key for key in previous_keys if key.sort_key in removed)
    return added_keys, removed_keys


class EffectiveCapabilitySet(BaseModel):
    """Deterministic query result — not runtime inventory authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["effective_capability_set.v1"] = SCHEMA_EFFECTIVE_CAPABILITY_SET_V1
    need: WorkStageCapabilityNeed
    governed_result: GovernedDiscoveryResult
    effective_candidates: tuple[GovernedCapabilityCandidate, ...]

    @model_validator(mode="after")
    def _validate_effective_candidates(self) -> EffectiveCapabilitySet:
        allowed = self.governed_result.allowed
        seen_identity_keys: set[tuple[str, str, str, str]] = set()
        for candidate in self.effective_candidates:
            if candidate.availability is not AvailabilityDisposition.HOST_AVAILABLE:
                raise ValueError(
                    "effective candidates must be HOST_AVAILABLE executable members",
                )
            if candidate not in allowed:
                raise ValueError(
                    "effective candidates must be members of governed_result.allowed",
                )
            identity_key = candidate.ranked.identity.sort_key
            if identity_key in seen_identity_keys:
                raise ValueError(
                    "effective candidates must not contain duplicate capability identities",
                )
            seen_identity_keys.add(identity_key)
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
    effective_set: EffectiveCapabilitySet

    @property
    def need(self) -> WorkStageCapabilityNeed:
        return self.effective_set.need

    @property
    def catalog_only_identity_keys(self) -> tuple[CapabilityIdentityKey, ...]:
        return _catalog_only_identity_keys(
            self.effective_set.governed_result.allowed,
            self.effective_set.effective_candidates,
        )


class WorkStageCapabilityTransitionEvidence(BaseModel):
    """Typed comparison between consecutive stage effective sets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["work_stage_capability_transition_evidence.v1"] = (
        SCHEMA_WORK_STAGE_CAPABILITY_TRANSITION_EVIDENCE_V1
    )
    previous: EffectiveCapabilitySet
    current: EffectiveCapabilitySet

    @model_validator(mode="after")
    def _validate_work_alignment(self) -> WorkStageCapabilityTransitionEvidence:
        if self.previous.need.work_reference != self.current.need.work_reference:
            raise ValueError("transition evidence requires the same work_reference")
        return self

    @property
    def added_identity_keys(self) -> tuple[CapabilityIdentityKey, ...]:
        added, _ = _transition_identity_diff(
            self.previous.effective_identity_keys,
            self.current.effective_identity_keys,
        )
        return added

    @property
    def removed_identity_keys(self) -> tuple[CapabilityIdentityKey, ...]:
        _, removed = _transition_identity_diff(
            self.previous.effective_identity_keys,
            self.current.effective_identity_keys,
        )
        return removed


def select_effective_executable_candidates(
    allowed: tuple[GovernedCapabilityCandidate, ...],
) -> tuple[GovernedCapabilityCandidate, ...]:
    """Narrow governed allowed candidates to HOST_AVAILABLE executable members."""
    executable = tuple(
        candidate
        for candidate in allowed
        if candidate.availability is AvailabilityDisposition.HOST_AVAILABLE
    )
    return tuple(
        sorted(
            executable,
            key=lambda candidate: candidate.ranked.identity.sort_key,
        )
    )


def compare_work_stage_effective_capabilities(
    previous: EffectiveCapabilitySet,
    current: EffectiveCapabilitySet,
) -> WorkStageCapabilityTransitionEvidence:
    """Derive added/removed identity keys between two stage results."""
    return WorkStageCapabilityTransitionEvidence(
        previous=previous,
        current=current,
    )
