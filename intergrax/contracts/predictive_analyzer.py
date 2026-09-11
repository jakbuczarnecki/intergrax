# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Predictive analyzer SPI — bounded risk signals only (PREDICTIVE R1)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.predictive_context import PredictiveContext
from intergrax.contracts.predictive_risk import PredictiveRiskSignal


@runtime_checkable
class PredictiveAnalyzer(Protocol):
    """Evidence-driven forward risk analysis — no Problem writes or root-cause claims."""

    @property
    def analyzer_id(self) -> str:
        """Stable analyzer identity for registry ordering and audit."""

    @property
    def analyzer_namespace(self) -> str:
        """Owning namespace (e.g. intergrax.platform)."""

    @property
    def priority(self) -> int:
        """Higher values run earlier; ties broken by namespace then analyzer_id."""

    @property
    def model_version(self) -> str:
        """Version string recorded on emitted signals."""

    def analyze(self, context: PredictiveContext) -> tuple[PredictiveRiskSignal, ...]:
        """Pure analysis over readonly context — no persistence or Problem authority."""


__all__ = ["PredictiveAnalyzer"]
