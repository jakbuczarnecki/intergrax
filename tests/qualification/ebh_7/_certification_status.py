# © Artur Czarnecki. All rights reserved.

"""EBH-7 certification status semantics (distinct from gate PASS labels)."""

from __future__ import annotations

from enum import StrEnum


class CertificationStatus(StrEnum):
    INVENTORIED = "INVENTORIED"
    PLANNED = "PLANNED"
    MATERIALIZED = "MATERIALIZED"
    QUALIFICATION_PENDING = "QUALIFICATION_PENDING"
    QUALIFIED = "QUALIFIED"
    BLOCKED = "BLOCKED"
    READY_FOR_INDEPENDENT_AUDIT = "READY_FOR_INDEPENDENT_AUDIT"
    INDEPENDENTLY_ACCEPTED = "INDEPENDENTLY_ACCEPTED"


def is_behavioral_certification(status: CertificationStatus) -> bool:
    return status in {
        CertificationStatus.QUALIFIED,
        CertificationStatus.READY_FOR_INDEPENDENT_AUDIT,
        CertificationStatus.INDEPENDENTLY_ACCEPTED,
    }
