# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Skill package identity contracts (domain-owned, vendor-neutral)."""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

_NON_EMPTY = Field(min_length=1)

SCHEMA_SKILL_PACKAGE_CANDIDATE_V1: Final = "skill_package_candidate.v1"
SCHEMA_SKILL_PACKAGE_IDENTITY_V1: Final = "skill_package_identity.v1"
SCHEMA_SKILL_DISCOVERY_CANDIDATE_IDENTITY_V1: Final = (
    "skill_discovery_candidate_identity.v1"
)


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class SkillPackageCandidate(BaseModel):
    """Exact package candidate selected from discovery — composition truth, not execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_SKILL_PACKAGE_CANDIDATE_V1
    logical_skill_id: str = _NON_EMPTY
    package_reference: str = _NON_EMPTY
    package_version: str = _NON_EMPTY
    package_digest: str | None = None

    @field_validator("logical_skill_id", "package_reference", "package_version")
    @classmethod
    def _strip_fields(cls, value: str) -> str:
        return _strip_required(value)

    @field_validator("package_digest")
    @classmethod
    def _strip_optional_digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)


class SkillPackageIdentity(BaseModel):
    """Digest-pinned package identity after exact resolution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_SKILL_PACKAGE_IDENTITY_V1
    logical_skill_id: str = _NON_EMPTY
    package_reference: str = _NON_EMPTY
    package_version: str = _NON_EMPTY
    package_digest: str = _NON_EMPTY

    @field_validator(
        "logical_skill_id",
        "package_reference",
        "package_version",
        "package_digest",
    )
    @classmethod
    def _strip_fields(cls, value: str) -> str:
        return _strip_required(value)

    @classmethod
    def from_candidate(cls, candidate: SkillPackageCandidate) -> SkillPackageIdentity:
        if candidate.package_digest is None:
            raise ValueError("package candidate lacks digest required for binding")
        return cls(
            logical_skill_id=candidate.logical_skill_id,
            package_reference=candidate.package_reference,
            package_version=candidate.package_version,
            package_digest=candidate.package_digest,
        )


class SkillDiscoveryCandidateIdentity(BaseModel):
    """Source-qualified federated skill candidate for acquisition resolution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_SKILL_DISCOVERY_CANDIDATE_IDENTITY_V1
    catalog_source_id: str = _NON_EMPTY
    package: SkillPackageCandidate

    @field_validator("catalog_source_id")
    @classmethod
    def _strip_source(cls, value: str) -> str:
        return _strip_required(value)


__all__ = [
    "SCHEMA_SKILL_DISCOVERY_CANDIDATE_IDENTITY_V1",
    "SCHEMA_SKILL_PACKAGE_CANDIDATE_V1",
    "SCHEMA_SKILL_PACKAGE_IDENTITY_V1",
    "SkillDiscoveryCandidateIdentity",
    "SkillPackageCandidate",
    "SkillPackageIdentity",
]
