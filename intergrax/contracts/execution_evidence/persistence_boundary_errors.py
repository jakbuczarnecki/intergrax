# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Persistence failure types visible at ``EvidencePersistencePort`` (NPSC-5F)."""

from __future__ import annotations

__all__ = [
    "EvidencePersistenceBoundaryError",
    "EvidencePersistenceIntegrityError",
    "MandatoryEvidencePersistenceError",
]


class EvidencePersistenceBoundaryError(Exception):
    """
    Execution evidence persistence failed at the port boundary.

    Storage/provider exceptions must be translated to this family before leaving
    the persistence adapter; execution producers depend only on these types.
    """


class EvidencePersistenceIntegrityError(EvidencePersistenceBoundaryError):
    """Integrity, idempotency, or tenant-routing violation at the persistence boundary."""


class MandatoryEvidencePersistenceError(EvidencePersistenceBoundaryError):
    """Mandatory execution evidence could not be durably committed."""
