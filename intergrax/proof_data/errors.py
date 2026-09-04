"""Proof data package distribution errors."""

from __future__ import annotations


class DataPackageError(RuntimeError):
    """Base error for proof data package operations."""


class DataPackageDescriptorError(DataPackageError):
    """Invalid or unsupported package descriptor."""


class DataPackageTransportError(DataPackageError):
    """Transport-layer failure obtaining package bytes."""


class DataPackageIntegrityError(DataPackageError):
    """Checksum, size, or file integrity validation failure."""


class DataPackageInstallError(DataPackageError):
    """Package installation orchestration failure."""
