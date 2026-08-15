# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical Python distribution package identity primitives."""

from __future__ import annotations

from packaging.utils import InvalidName, canonicalize_name
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict, field_validator


def _require_non_empty_text(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def normalize_distribution_package_name(value: str) -> str:
    """Normalize and validate a Python distribution package name."""
    normalized = _require_non_empty_text(value, field_name="package name")
    try:
        return canonicalize_name(normalized, validate=True)
    except InvalidName as exc:
        raise ValueError(f"invalid package name: {normalized!r}") from exc


def normalize_package_version(value: str) -> str:
    """Normalize and validate a PEP 440 package version."""
    normalized = _require_non_empty_text(value, field_name="package version")
    try:
        return str(Version(normalized))
    except InvalidVersion as exc:
        raise ValueError(f"invalid package version: {normalized!r}") from exc


class DistributionPackageIdentity(BaseModel):
    """Canonical distribution package identity (name + version)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    version: str

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return normalize_distribution_package_name(value)

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: str) -> str:
        return normalize_package_version(value)


def package_identities_conflict(
    left: DistributionPackageIdentity,
    right: DistributionPackageIdentity,
) -> bool:
    """Return whether two package identity declarations conflict on name/version."""
    return left.name == right.name and left.version != right.version
