# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Persistence port for autonomy qualification runs (SELF-HEALING R6.4)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.autonomy.qualification.result import AutonomyQualificationResult


@runtime_checkable
class AutonomyQualificationRepository(Protocol):
    def save(self, result: AutonomyQualificationResult) -> None:
        """Persist qualification outcome for audit replay — adapter supplied by host."""
        ...

    def get(self, validation_id: str) -> AutonomyQualificationResult | None: ...


__all__ = ["AutonomyQualificationRepository"]
