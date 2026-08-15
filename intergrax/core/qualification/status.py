# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical qualification status vocabulary and ordering."""

from __future__ import annotations

from enum import StrEnum


class QualificationStatus(StrEnum):
    """Qualification outcome — distinct from lifecycle/discovery states."""

    NOT_QUALIFIED = "not_qualified"
    QUALIFIED = "qualified"
    PRODUCTION_QUALIFIED = "production_qualified"
    REJECTED = "rejected"


_QUALIFICATION_STATUS_RANK: dict[QualificationStatus, int] = {
    QualificationStatus.NOT_QUALIFIED: 0,
    QualificationStatus.REJECTED: 0,
    QualificationStatus.QUALIFIED: 1,
    QualificationStatus.PRODUCTION_QUALIFIED: 2,
}


def qualification_status_satisfies(
    actual: QualificationStatus,
    required: QualificationStatus,
) -> bool:
    """Return whether ``actual`` meets or exceeds ``required`` qualification."""
    return _QUALIFICATION_STATUS_RANK[actual] >= _QUALIFICATION_STATUS_RANK[required]
