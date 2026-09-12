# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Descriptive action intent for autonomy evaluation — not an execution command (R6.1)."""

from __future__ import annotations

from enum import StrEnum


class AutonomyRequestedActionKind(StrEnum):
    """Intent labels for classification only — no executor or workflow binding."""

    CONSIDER_STRATEGY_FOR_EXECUTION = "CONSIDER_STRATEGY_FOR_EXECUTION"
    CLASSIFY_POSTURE_ONLY = "CLASSIFY_POSTURE_ONLY"


__all__ = ["AutonomyRequestedActionKind"]
