# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Autonomy safety validator contract (SELF-HEALING R6.4)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.autonomy.qualification.context import AutonomySafetyQualificationContext
from intergrax.contracts.self_healing.autonomy.qualification.result import AutonomyQualificationResult


@runtime_checkable
class AutonomySafetyValidator(Protocol):
    """Qualifies autonomy configuration safety without performing actions."""

    def qualify(self, context: AutonomySafetyQualificationContext) -> AutonomyQualificationResult: ...


__all__ = ["AutonomySafetyValidator"]
