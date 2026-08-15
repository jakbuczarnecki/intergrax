# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical distribution and platform compatibility primitives."""

from intergrax.core.distribution.errors import (
    DistributionError,
    InvalidPlatformVersionError,
    PlatformIncompatibilityError,
)
from intergrax.core.distribution.package_identity import (
    DistributionPackageIdentity,
    normalize_distribution_package_name,
    normalize_package_version,
    package_identities_conflict,
)
from intergrax.core.distribution.platform_compatibility import (
    PlatformCompatibility,
    PlatformCompatibilityReason,
    PlatformCompatibilityResult,
    check_platform_compatibility,
    normalize_platform_version,
    require_platform_compatibility,
)

__all__ = [
    "DistributionError",
    "DistributionPackageIdentity",
    "InvalidPlatformVersionError",
    "PlatformCompatibility",
    "PlatformCompatibilityReason",
    "PlatformCompatibilityResult",
    "PlatformIncompatibilityError",
    "check_platform_compatibility",
    "normalize_distribution_package_name",
    "normalize_package_version",
    "normalize_platform_version",
    "package_identities_conflict",
    "require_platform_compatibility",
]
