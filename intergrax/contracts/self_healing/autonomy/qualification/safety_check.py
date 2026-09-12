# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Safety check plugin contract (SELF-HEALING R6.4)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.autonomy.qualification.context import AutonomySafetyQualificationContext
from intergrax.contracts.self_healing.autonomy.qualification.result import AutonomySafetyCheckResult


@runtime_checkable
class AutonomySafetyCheck(Protocol):
    """Single aspect of autonomy configuration safety — read-only inspection."""

    @property
    def check_id(self) -> str: ...

    def run(self, context: AutonomySafetyQualificationContext) -> AutonomySafetyCheckResult:
        """Must not invoke executors, mutate configuration, or change runtime state."""
        ...


__all__ = ["AutonomySafetyCheck"]
