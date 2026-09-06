# © Artur Czarnecki. All rights reserved.

"""Inspection result completeness — not capability readiness (P1.4)."""

from __future__ import annotations

from enum import StrEnum


class InspectionCompleteness(StrEnum):
    """How complete an inspection read-model projection is."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
