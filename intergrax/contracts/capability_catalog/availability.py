# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Availability disposition vocabulary for discovery projections (Stage 3)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class AvailabilityDisposition(StrEnum):
    """Read-only availability projection — not governance authority.

    Stage 3 surfaces disposition from caller-supplied evidence and catalog
    membership. It does not evaluate policy or grant permissions.
    """

    CATALOG_AVAILABLE = "catalog_available"
    HOST_AVAILABLE = "host_available"
    BLOCKED = "blocked"
    UNAVAILABLE = "unavailable"
    SCOPE_UNAVAILABLE = "scope_unavailable"


NORMATIVE_AVAILABILITY_DISPOSITIONS: Final[frozenset[AvailabilityDisposition]] = frozenset(
    {
        AvailabilityDisposition.CATALOG_AVAILABLE,
        AvailabilityDisposition.HOST_AVAILABLE,
        AvailabilityDisposition.BLOCKED,
        AvailabilityDisposition.UNAVAILABLE,
        AvailabilityDisposition.SCOPE_UNAVAILABLE,
    }
)
