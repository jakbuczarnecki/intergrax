# © Artur Czarnecki. All rights reserved.

"""Structured inspection explainability contracts (P1.4)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from intergrax.applications.contracts.profile_resolution.revision_id import (
    EffectiveProfileRevisionId,
)
from intergrax.contracts.execution_identity import ExecutionId


class InspectionProvenanceRef(BaseModel):
    """Typed provenance pointer to canonical facts — not authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str = Field(min_length=1)
    ref: str = Field(min_length=1)


class InspectionExplanation(BaseModel):
    """Deterministic structured explanation derived from canonical facts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    subject: str = Field(min_length=1)
    facts: tuple[str, ...] = Field(default_factory=tuple)
    reasons: tuple[str, ...] = Field(default_factory=tuple)
    provenance_refs: tuple[InspectionProvenanceRef, ...] = Field(default_factory=tuple)
    related_revision_id: EffectiveProfileRevisionId | None = None
    related_execution_id: ExecutionId | None = None
