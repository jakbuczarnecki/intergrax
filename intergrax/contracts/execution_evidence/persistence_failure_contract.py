# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Controlled persistence failure model visible to Execution Runtime (not providers)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "ControlledEvidencePersistenceFailure",
    "EvidencePersistenceFailureCategory",
]


class EvidencePersistenceFailureCategory(str, Enum):
    """Provider-agnostic persistence problem category at the evidence port."""

    INTEGRITY = "integrity"
    INFRASTRUCTURE = "infrastructure"


@dataclass(frozen=True, slots=True)
class ControlledEvidencePersistenceFailure:
    """
    Runtime-facing view of a persistence port failure.

    Describes whether execution may continue after the failure without exposing
    storage or provider implementation details.
    """

    category: EvidencePersistenceFailureCategory
    runtime_may_continue: bool
