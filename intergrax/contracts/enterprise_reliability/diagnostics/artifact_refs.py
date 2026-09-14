# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Opaque authoritative ERL artifact references for diagnostics — refs only, no payloads."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ReliabilityDiagnosticArtifactRefs(BaseModel):
    """Links to committed ERL artifacts at emission time."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_ref: str | None = Field(default=None, max_length=512)
    reconciliation_ref: str | None = Field(default=None, max_length=512)
    resolution_ref: str | None = Field(default=None, max_length=512)
    governance_ref: str | None = Field(default=None, max_length=512)
    recovery_ref: str | None = Field(default=None, max_length=512)
    source_transition_ref: str | None = Field(default=None, max_length=512)


__all__ = ["ReliabilityDiagnosticArtifactRefs"]
