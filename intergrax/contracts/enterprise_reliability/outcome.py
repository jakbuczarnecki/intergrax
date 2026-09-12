# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical external-effect outcome tri-state (ERL Phase 1)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

SCHEMA_EXTERNAL_EFFECT_OUTCOME_V1: Final = "external_effect_outcome.v1"


class ExternalEffectOutcome(StrEnum):
    """Operational truth for an external effect — not a Reliability failure class."""

    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"


_TERMINAL_OUTCOMES = frozenset(
    {
        ExternalEffectOutcome.SUCCESS,
        ExternalEffectOutcome.FAILURE,
    }
)


def is_terminal_external_effect_outcome(outcome: ExternalEffectOutcome) -> bool:
    """Return whether the outcome is definitively terminal (not UNKNOWN)."""
    return outcome in _TERMINAL_OUTCOMES


__all__ = [
    "ExternalEffectOutcome",
    "SCHEMA_EXTERNAL_EFFECT_OUTCOME_V1",
    "is_terminal_external_effect_outcome",
]
