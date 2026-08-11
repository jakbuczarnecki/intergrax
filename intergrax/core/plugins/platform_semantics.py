# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared Platform Plugin lifecycle, compatibility, and conflict vocabulary (PLATFORM-PLUGIN-6)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from packaging.version import InvalidVersion, Version

from intergrax.core.plugins.errors import (
    InvalidPlatformVersionError,
    PlatformIncompatibilityError,
)
from intergrax.core.plugins.package_contract import (
    PlatformCompatibility,
    PlatformPluginManifest,
    PluginPackageIdentity,
)


class PlatformPluginLifecycleState(StrEnum):
    """Cross-cutting lifecycle vocabulary (observability and coordination only)."""

    DISCOVERED = "discovered"
    VALIDATED = "validated"
    ENABLED = "enabled"
    MATERIALIZED = "materialized"
    ACTIVE = "active"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class PlatformPluginConflictKind(StrEnum):
    """Shared conflict classification; resolution policy remains domain-owned."""

    PACKAGE_IDENTITY = "package_identity"
    ENTRY_POINT_NAME = "entry_point_name"
    CAPABILITY_IDENTITY = "capability_identity"
    DOMAIN_RESOURCE_ID = "domain_resource_id"


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


def check_manifest_platform_compatibility(
    manifest: PlatformPluginManifest,
    platform_version: str | Version,
) -> PlatformCompatibilityResult:
    """Check package-level platform compatibility for a manifest."""
    return check_platform_compatibility(manifest.platform_compatibility, platform_version)


def package_identities_conflict(
    left: PluginPackageIdentity,
    right: PluginPackageIdentity,
) -> bool:
    """Return whether two package identity declarations conflict on name/version."""
    return left.name == right.name and left.version != right.version
