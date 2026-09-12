# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Autonomy level model — enterprise posture tiers (SELF-HEALING R6.1)."""

from __future__ import annotations

from enum import StrEnum


class AutonomyLevel(StrEnum):
    """
    Describes how much autonomous action is permitted under enterprise rules.

    ``FULL_AUTONOMY`` exists only as a forward-compatible contract value and
    must never be selected as an active runtime level in R6 implementations.
    """

    OBSERVE_ONLY = "OBSERVE_ONLY"
    RECOMMEND_ONLY = "RECOMMEND_ONLY"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    CONTROLLED_EXECUTION = "CONTROLLED_EXECUTION"
    FULL_AUTONOMY = "FULL_AUTONOMY"

    def ensure_runtime_activatable(self) -> None:
        """Reject levels that must not be activated in the current program phase."""
        if self is AutonomyLevel.FULL_AUTONOMY:
            raise ValueError("FULL_AUTONOMY is contract-only and cannot be activated")


__all__ = ["AutonomyLevel"]
