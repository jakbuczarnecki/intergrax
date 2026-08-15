# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical Intergrax platform compatibility primitives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.core.distribution.errors import (
    InvalidPlatformVersionError,
    PlatformIncompatibilityError,
)
from intergrax.core.distribution.package_identity import _require_non_empty_text


class PlatformCompatibility(BaseModel):
    """Declared Intergrax platform compatibility metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    intergrax_version: str = Field(
        description="Declared Intergrax platform version specifier range.",
    )

    @field_validator("intergrax_version")
    @classmethod
    def _validate_intergrax_version(cls, value: str) -> str:
        normalized = _require_non_empty_text(value, field_name="intergrax_version")
        try:
            return str(SpecifierSet(normalized))
        except InvalidSpecifier as exc:
            raise ValueError(f"invalid intergrax_version specifier: {normalized!r}") from exc

    @property
    def declared_specifier(self) -> SpecifierSet:
        """Return the declared compatibility specifier (metadata only)."""
        return SpecifierSet(self.intergrax_version)


class PlatformCompatibilityReason(StrEnum):
    """Auditable compatibility outcome codes (compatible does not imply qualified)."""

    COMPATIBLE = "compatible"
    INCOMPATIBLE_VERSION = "incompatible_version"
    INVALID_PLATFORM_VERSION = "invalid_platform_version"


@dataclass(frozen=True, slots=True)
class PlatformCompatibilityResult:
    """Deterministic platform compatibility check result."""

    declared_specifier: str
    tested_platform_version: str
    compatible: bool
    reason: PlatformCompatibilityReason

    def __post_init__(self) -> None:
        if self.compatible and self.reason is not PlatformCompatibilityReason.COMPATIBLE:
            raise ValueError("compatible result requires reason COMPATIBLE")
        if not self.compatible and self.reason is PlatformCompatibilityReason.COMPATIBLE:
            raise ValueError("incompatible result cannot use reason COMPATIBLE")


def normalize_platform_version(platform_version: str | Version) -> str:
    """Normalize a platform version string for specifier evaluation."""
    if isinstance(platform_version, Version):
        return str(platform_version)
    normalized = platform_version.strip()
    if not normalized:
        raise InvalidPlatformVersionError("platform version must be non-empty")
    try:
        return str(Version(normalized))
    except InvalidVersion as exc:
        raise InvalidPlatformVersionError(
            f"invalid platform version: {platform_version!r}"
        ) from exc


def check_platform_compatibility(
    declared: PlatformCompatibility,
    platform_version: str | Version,
) -> PlatformCompatibilityResult:
    """Return whether ``declared`` is compatible with ``platform_version``.

    Does not mutate ``declared`` and does not imply qualification or trust.
    """
    declared_specifier = str(declared.declared_specifier)
    try:
        tested_version = normalize_platform_version(platform_version)
    except InvalidPlatformVersionError:
        return PlatformCompatibilityResult(
            declared_specifier=declared_specifier,
            tested_platform_version=str(platform_version),
            compatible=False,
            reason=PlatformCompatibilityReason.INVALID_PLATFORM_VERSION,
        )
    compatible = tested_version in declared.declared_specifier
    reason = (
        PlatformCompatibilityReason.COMPATIBLE
        if compatible
        else PlatformCompatibilityReason.INCOMPATIBLE_VERSION
    )
    return PlatformCompatibilityResult(
        declared_specifier=declared_specifier,
        tested_platform_version=tested_version,
        compatible=compatible,
        reason=reason,
    )


def require_platform_compatibility(
    declared: PlatformCompatibility,
    platform_version: str | Version,
) -> PlatformCompatibilityResult:
    """Raise ``PlatformIncompatibilityError`` when the declared range does not match."""
    result = check_platform_compatibility(declared, platform_version)
    if result.compatible:
        return result
    if result.reason is PlatformCompatibilityReason.INVALID_PLATFORM_VERSION:
        raise InvalidPlatformVersionError(
            f"invalid platform version for compatibility check: {platform_version!r}"
        )
    raise PlatformIncompatibilityError(
        (
            "platform plugin package is incompatible with Intergrax platform version "
            f"{result.tested_platform_version!r}; declared range is "
            f"{result.declared_specifier!r}"
        ),
        result=result,
    )
