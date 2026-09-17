# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Canonical runtime inspection completeness (INSPECT-01-A)."""

from __future__ import annotations

from enum import StrEnum


class RuntimeInspectionCompleteness(StrEnum):
    """How complete a federated runtime inspection snapshot is for configured sources."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    REDACTED = "redacted"


__all__ = ["RuntimeInspectionCompleteness"]
