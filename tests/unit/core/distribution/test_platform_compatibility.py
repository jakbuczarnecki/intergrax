# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.core.distribution import (
    InvalidPlatformVersionError,
    PlatformCompatibility,
    PlatformCompatibilityReason,
    PlatformIncompatibilityError,
    check_platform_compatibility,
    normalize_platform_version,
    require_platform_compatibility,
)

pytestmark = pytest.mark.unit


def test_platform_compatibility_unchanged_behavior() -> None:
    declared = PlatformCompatibility(intergrax_version=">=1,<2")
    result = check_platform_compatibility(declared, "1.5")
    assert result.compatible is True
    assert result.reason is PlatformCompatibilityReason.COMPATIBLE

    excluded = check_platform_compatibility(declared, "2.0")
    assert excluded.compatible is False
    assert excluded.reason is PlatformCompatibilityReason.INCOMPATIBLE_VERSION


def test_normalize_platform_version_rejects_empty() -> None:
    with pytest.raises(InvalidPlatformVersionError):
        normalize_platform_version("   ")


def test_require_platform_compatibility_raises_on_mismatch() -> None:
    declared = PlatformCompatibility(intergrax_version=">=2")
    with pytest.raises(PlatformIncompatibilityError):
        require_platform_compatibility(declared, "1.0")
