# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Capability realization outcome vocabulary (UCA-2)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class CapabilityRealizationOutcome(StrEnum):
    """Terminal realization coordination outcome — not execution or governance grant."""

    SUCCEEDED = "succeeded"
    NOT_SUPPORTED = "not_supported"
    BLOCKED = "blocked"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    REQUIRES_HITL = "requires_hitl"
    CONFLICT = "conflict"


NORMATIVE_CAPABILITY_REALIZATION_OUTCOMES: Final[
    frozenset[CapabilityRealizationOutcome]
] = frozenset(CapabilityRealizationOutcome)


__all__ = [
    "CapabilityRealizationOutcome",
    "NORMATIVE_CAPABILITY_REALIZATION_OUTCOMES",
]
