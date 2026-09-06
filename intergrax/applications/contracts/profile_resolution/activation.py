# © Artur Czarnecki. All rights reserved.

"""Atomic effective profile revision activation contracts (P1.6)."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.applications.contracts.profile_resolution.revision import (
    EffectiveProfileRevisionScope,
)
from intergrax.applications.contracts.profile_resolution.revision_id import (
    EffectiveProfileRevisionId,
    validate_effective_profile_revision_id,
)


class ActiveEffectiveProfileRevisionBinding(BaseModel):
    """Immutable scoped active revision pointer — read contract, not revision authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "active_effective_profile_revision_binding.v1"
    scope: EffectiveProfileRevisionScope
    revision_id: EffectiveProfileRevisionId
    fingerprint: str = Field(min_length=1)

    @field_validator("revision_id", mode="before")
    @classmethod
    def _validate_revision_id(cls, value: object) -> EffectiveProfileRevisionId:
        if isinstance(value, EffectiveProfileRevisionId):
            return value
        return EffectiveProfileRevisionId(validate_effective_profile_revision_id(value))


class ActiveEffectiveProfileRevisionCasOutcome(StrEnum):
    """Compare-and-set publication outcome for one scope."""

    UPDATED = "updated"
    UNCHANGED = "unchanged"
    CONFLICT = "conflict"


class ActiveEffectiveProfileRevisionCasResult(BaseModel):
    """Typed CAS result — never ambiguous bool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: ActiveEffectiveProfileRevisionCasOutcome
    current_binding: ActiveEffectiveProfileRevisionBinding | None = None


class ActivateEffectiveProfileRevisionRequest(BaseModel):
    """Activation input with expected-current CAS semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scope: EffectiveProfileRevisionScope
    candidate_revision_id: EffectiveProfileRevisionId
    expected_active_revision_id: EffectiveProfileRevisionId | None = None

    @field_validator("candidate_revision_id", mode="before")
    @classmethod
    def _validate_candidate_revision_id(cls, value: object) -> EffectiveProfileRevisionId:
        if isinstance(value, EffectiveProfileRevisionId):
            return value
        return EffectiveProfileRevisionId(validate_effective_profile_revision_id(value))

    @field_validator("expected_active_revision_id", mode="before")
    @classmethod
    def _validate_expected_revision_id(
        cls,
        value: object,
    ) -> EffectiveProfileRevisionId | None:
        if value is None:
            return None
        if isinstance(value, EffectiveProfileRevisionId):
            return value
        return EffectiveProfileRevisionId(validate_effective_profile_revision_id(value))


class EffectiveProfileActivationResult(BaseModel):
    """Post-commit activation evidence — observability-safe receipt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "effective_profile_activation.v1"
    scope: EffectiveProfileRevisionScope
    previous_revision_id: EffectiveProfileRevisionId | None = None
    active_revision_id: EffectiveProfileRevisionId
    active_fingerprint: str = Field(min_length=1)
    changed: bool

    @field_validator("previous_revision_id", mode="before")
    @classmethod
    def _validate_previous_revision_id(
        cls,
        value: object,
    ) -> EffectiveProfileRevisionId | None:
        if value is None:
            return None
        if isinstance(value, EffectiveProfileRevisionId):
            return value
        return EffectiveProfileRevisionId(validate_effective_profile_revision_id(value))

    @field_validator("active_revision_id", mode="before")
    @classmethod
    def _validate_active_revision_id(cls, value: object) -> EffectiveProfileRevisionId:
        if isinstance(value, EffectiveProfileRevisionId):
            return value
        return EffectiveProfileRevisionId(validate_effective_profile_revision_id(value))


class ActiveEffectiveProfileRevisionStore(Protocol):
    """Scoped atomic active revision pointer persistence."""

    def get_active(
        self,
        scope: EffectiveProfileRevisionScope,
    ) -> ActiveEffectiveProfileRevisionBinding | None:
        """Return one immutable active binding snapshot for scope."""

    def compare_and_set_active(
        self,
        scope: EffectiveProfileRevisionScope,
        *,
        expected_revision_id: EffectiveProfileRevisionId | None,
        new_binding: ActiveEffectiveProfileRevisionBinding,
    ) -> ActiveEffectiveProfileRevisionCasResult:
        """Atomically publish active binding when expected current matches."""
