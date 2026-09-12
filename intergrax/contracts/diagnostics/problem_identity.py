# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Stable diagnostic Problem identity types (DIAG-5D / HARDENING-8)."""

from __future__ import annotations

from enum import StrEnum
from typing import NewType

ProblemId = NewType("ProblemId", str)


class ProblemStatus(StrEnum):
    OPEN = "open"
    RESOLVED = "resolved"


class ProblemOccurrenceAggregateHealth(StrEnum):
    """Operator-readable aggregate projection quality for one Problem."""

    CONSISTENT = "consistent"
    RECONCILIATION_REQUIRED = "reconciliation_required"


__all__ = [
    "ProblemId",
    "ProblemOccurrenceAggregateHealth",
    "ProblemStatus",
]
