# © Artur Czarnecki. All rights reserved.

"""Effective profile semantic diff contracts (P1.2)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from intergrax.applications.contracts.profile_resolution.revision_id import (
    EffectiveProfileRevisionId,
)


class ProfileDiffChangeKind(StrEnum):
    """Typed semantic change classification."""

    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"
    NARROWED = "narrowed"
    WIDENED = "widened"


class ProfileDiffProvenanceRef(BaseModel):
    """Typed reference to resolution evidence without copying full history."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    decision_index: int = Field(ge=0)


class ProfileDiffEntry(BaseModel):
    """One domain-aware semantic diff entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    before: str | None
    after: str | None
    change_kind: ProfileDiffChangeKind
    provenance: tuple[ProfileDiffProvenanceRef, ...] = Field(default_factory=tuple)


class EffectiveProfileDiff(BaseModel):
    """Machine-readable semantic diff between two effective revisions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "effective_profile_diff.v1"
    from_revision_id: EffectiveProfileRevisionId
    to_revision_id: EffectiveProfileRevisionId
    from_fingerprint: str
    to_fingerprint: str
    entries: tuple[ProfileDiffEntry, ...] = Field(default_factory=tuple)

    @property
    def is_empty(self) -> bool:
        return not self.entries
