# © Artur Czarnecki. All rights reserved.

"""Controlled failures for production decision orchestration (DS-E2E-15J-L6)."""

from __future__ import annotations


class DecisionOrchestrationError(Exception):
    """Base error for decision orchestration boundary failures."""


class DecisionOrchestrationProviderMissingError(DecisionOrchestrationError):
    """Raised when a required orchestration provider was not supplied."""


__all__ = [
    "DecisionOrchestrationError",
    "DecisionOrchestrationProviderMissingError",
]
