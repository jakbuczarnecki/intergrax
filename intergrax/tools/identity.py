# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tool package identity contracts (domain-owned, vendor-neutral)."""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

_NON_EMPTY = Field(min_length=1)

SCHEMA_TOOL_PACKAGE_CANDIDATE_V1: Final = "tool_package_candidate.v1"
SCHEMA_TOOL_PACKAGE_IDENTITY_V1: Final = "tool_package_identity.v1"
SCHEMA_TOOL_DISCOVERY_CANDIDATE_IDENTITY_V1: Final = (
    "tool_discovery_candidate_identity.v1"
)


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class ToolPackageCandidate(BaseModel):
    """Exact package candidate selected from discovery — not execution truth alone."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_TOOL_PACKAGE_CANDIDATE_V1
    logical_tool_id: str = _NON_EMPTY
    package_reference: str = _NON_EMPTY
    package_version: str = _NON_EMPTY
    package_digest: str | None = None

    @field_validator("logical_tool_id", "package_reference", "package_version")
    @classmethod
    def _strip_fields(cls, value: str) -> str:
        return _strip_required(value)

    @field_validator("package_digest")
    @classmethod
    def _strip_optional_digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)


class ToolPackageIdentity(BaseModel):
    """Digest-pinned package identity after exact resolution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_TOOL_PACKAGE_IDENTITY_V1
    logical_tool_id: str = _NON_EMPTY
    package_reference: str = _NON_EMPTY
    package_version: str = _NON_EMPTY
    package_digest: str = _NON_EMPTY

    @field_validator(
        "logical_tool_id",
        "package_reference",
        "package_version",
        "package_digest",
    )
    @classmethod
    def _strip_fields(cls, value: str) -> str:
        return _strip_required(value)

    @classmethod
    def from_candidate(cls, candidate: ToolPackageCandidate) -> ToolPackageIdentity:
        if candidate.package_digest is None:
            raise ValueError("package candidate lacks digest required for activation")
        return cls(
            logical_tool_id=candidate.logical_tool_id,
            package_reference=candidate.package_reference,
            package_version=candidate.package_version,
            package_digest=candidate.package_digest,
        )


class ToolDiscoveryCandidateIdentity(BaseModel):
    """Source-qualified federated tool candidate for acquisition resolution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_TOOL_DISCOVERY_CANDIDATE_IDENTITY_V1
    catalog_source_id: str = _NON_EMPTY
    package: ToolPackageCandidate

    @field_validator("catalog_source_id")
    @classmethod
    def _strip_source(cls, value: str) -> str:
        return _strip_required(value)


__all__ = [
    "SCHEMA_TOOL_DISCOVERY_CANDIDATE_IDENTITY_V1",
    "SCHEMA_TOOL_PACKAGE_CANDIDATE_V1",
    "SCHEMA_TOOL_PACKAGE_IDENTITY_V1",
    "ToolDiscoveryCandidateIdentity",
    "ToolPackageCandidate",
    "ToolPackageIdentity",
]
