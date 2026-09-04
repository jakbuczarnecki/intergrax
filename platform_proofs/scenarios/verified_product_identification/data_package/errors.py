"""VPI data package errors."""

from __future__ import annotations


class VpiDataPackageError(RuntimeError):
    """Base error for VPI data package operations."""


class VpiDataPackageConfigurationError(VpiDataPackageError):
    """Invalid VPI data package configuration."""


class VpiDataPackageCompatibilityError(VpiDataPackageError):
    """Installed package is incompatible with current VPI identity."""


class VpiDataPackageNotInstalledError(VpiDataPackageError):
    """Required VPI data package is missing."""
