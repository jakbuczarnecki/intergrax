# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Errors for canonical distribution and platform compatibility primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from intergrax.core.distribution.platform_compatibility import PlatformCompatibilityResult


class DistributionError(Exception):
    """Base error for distribution primitives."""


class InvalidPlatformVersionError(DistributionError):
    """Platform version string is not a valid PEP 440 version."""


class PlatformIncompatibilityError(DistributionError):
    """Declared platform compatibility range does not include the tested version."""

    def __init__(self, message: str, *, result: PlatformCompatibilityResult | None = None) -> None:
        super().__init__(message)
        self.result = result
