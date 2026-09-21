# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Capability qualification outcome vocabulary (UCA-4)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class CapabilityQualificationOutcome(StrEnum):
    """Terminal qualification outcome — not execution or host availability."""

    QUALIFIED = "qualified"
    REJECTED = "rejected"
    BLOCKED = "blocked"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    REQUIRES_HITL = "requires_hitl"
    CONFLICT = "conflict"
    NOT_SUPPORTED = "not_supported"
    NO_PROVIDER = "no_provider"


NORMATIVE_CAPABILITY_QUALIFICATION_OUTCOMES: Final[
    frozenset[CapabilityQualificationOutcome]
] = frozenset(CapabilityQualificationOutcome)


__all__ = [
    "CapabilityQualificationOutcome",
    "NORMATIVE_CAPABILITY_QUALIFICATION_OUTCOMES",
]
