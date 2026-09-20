# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Capability acquisition outcome vocabulary (UCA-3)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class CapabilityAcquisitionOutcome(StrEnum):
    """Terminal acquisition coordination outcome — not qualification or execution."""

    SUCCEEDED = "succeeded"
    NOT_SUPPORTED = "not_supported"
    NO_STRATEGY = "no_strategy"
    CONFLICT = "conflict"
    BLOCKED = "blocked"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    REQUIRES_HITL = "requires_hitl"


NORMATIVE_CAPABILITY_ACQUISITION_OUTCOMES: Final[
    frozenset[CapabilityAcquisitionOutcome]
] = frozenset(CapabilityAcquisitionOutcome)


__all__ = [
    "CapabilityAcquisitionOutcome",
    "NORMATIVE_CAPABILITY_ACQUISITION_OUTCOMES",
]
