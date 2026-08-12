# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Agent package and distribution identity contracts (AGENT_DISTRIBUTION §6)."""

from __future__ import annotations

import re
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.agent_distribution._digest import (
    normalize_optional_package_digest,
    normalize_package_digest,
)

_NON_EMPTY = Field(min_length=1)
_PACKAGE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*(-[a-z0-9_]+)*$")

SCHEMA_AGENT_PACKAGE_IDENTITY_V1: Final = "agent_package_identity.v1"
SCHEMA_AGENT_PACKAGE_CANDIDATE_V1: Final = "agent_package_candidate.v1"


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


def _normalize_package_name(value: str) -> str:
    normalized = _strip_required(value)
    if not _PACKAGE_NAME_RE.match(normalized):
        raise ValueError(
            "distribution_package_id must be a normalized PyPI-style package name"
        )
    return normalized


class AgentPackageCandidate(BaseModel):
    """Pre-verification package identity from catalog resolution (digest optional)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_AGENT_PACKAGE_CANDIDATE_V1
    distribution_package_id: str = _NON_EMPTY
    package_version: str = _NON_EMPTY
    package_digest: str | None = None
    artifact_locator: str | None = None
    contract_id: str | None = None
    platform_compatibility_spec: str | None = None
    python_requires: str | None = None

    @field_validator("distribution_package_id")
    @classmethod
    def _validate_package_id(cls, value: str) -> str:
        return _normalize_package_name(value)

    @field_validator("package_version", "artifact_locator", "contract_id", "platform_compatibility_spec", "python_requires")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)

    @field_validator("package_digest")
    @classmethod
    def _validate_optional_digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalize_optional_package_digest(value)

    def to_digest_pinned(self) -> AgentPackageIdentity:
        """Promote to digest-pinned identity after artifact verification."""
        if self.package_digest is None:
            raise ValueError("package_digest is required for digest-pinned identity")
        return AgentPackageIdentity(
            distribution_package_id=self.distribution_package_id,
            package_version=self.package_version,
            package_digest=self.package_digest,
            artifact_locator=self.artifact_locator,
            contract_id=self.contract_id,
            platform_compatibility_spec=self.platform_compatibility_spec,
            python_requires=self.python_requires,
        )


class AgentPackageIdentity(BaseModel):
    """Canonical digest-pinned package identity — production authority (§6.2)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_AGENT_PACKAGE_IDENTITY_V1
    distribution_package_id: str = _NON_EMPTY
    package_version: str = _NON_EMPTY
    package_digest: str = _NON_EMPTY
    artifact_locator: str | None = None
    contract_id: str | None = None
    platform_compatibility_spec: str | None = None
    python_requires: str | None = None

    @field_validator("distribution_package_id")
    @classmethod
    def _validate_package_id(cls, value: str) -> str:
        return _normalize_package_name(value)

    @field_validator("package_version")
    @classmethod
    def _validate_version(cls, value: str) -> str:
        return _strip_required(value)

    @field_validator("package_digest")
    @classmethod
    def _validate_digest(cls, value: str) -> str:
        return normalize_package_digest(value)

    @field_validator("artifact_locator", "contract_id", "platform_compatibility_spec", "python_requires")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)
