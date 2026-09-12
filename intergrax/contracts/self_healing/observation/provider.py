# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Post-execution observation SPI (SELF-HEALING R3)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from intergrax.contracts.self_healing.execution.context import SelfHealingExecutionContext


@dataclass(frozen=True, slots=True)
class ObservationResult:
    metrics: tuple[tuple[str, str], ...]
    evidence_refs: tuple[str, ...]
    confidence: float
    timestamp: datetime

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0.0, 1.0]")
        if not self.evidence_refs:
            raise ValueError("observation requires evidence_refs")


@runtime_checkable
class SelfHealingObservationProvider(Protocol):
    @property
    def provider_id(self) -> str: ...

    def observe(
        self,
        execution_context: SelfHealingExecutionContext,
        *,
        before_metrics: tuple[tuple[str, str], ...] = (),
    ) -> ObservationResult:
        """Collect evidence-backed metrics after spine execution — no mutations."""


__all__ = ["ObservationResult", "SelfHealingObservationProvider"]
