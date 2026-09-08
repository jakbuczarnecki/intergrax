# © Artur Czarnecki. All rights reserved.

"""Evidence-side typed reference to immutable WorkArtifactVersion (MP-3G).

Evidence / Proof references Collaborative Work artifact versions without
ownership transfer or mutable lineage on the version record.
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.collaborative_work import WorkArtifactVersionRef

SCHEMA_EVIDENCE_ARTIFACT_VERSION_LINK_V1: Final = "evidence_artifact_version_link.v1"

_NON_EMPTY = Field(min_length=1)


class EvidenceArtifactVersionLink(BaseModel):
    """Typed association: evidence record references one WorkArtifactVersion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["evidence_artifact_version_link.v1"] = SCHEMA_EVIDENCE_ARTIFACT_VERSION_LINK_V1
    evidence_id: str = _NON_EMPTY
    artifact_version: WorkArtifactVersionRef

    @field_validator("evidence_id")
    @classmethod
    def _strip_evidence_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized
