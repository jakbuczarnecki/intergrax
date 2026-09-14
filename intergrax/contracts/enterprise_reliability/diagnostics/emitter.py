# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Best-effort emission port for ERL reliability diagnostics — no execution authority."""

from __future__ import annotations

from typing import Protocol

from intergrax.contracts.enterprise_reliability.diagnostics.observation import (
    ExternalEffectReliabilityObservation,
)


class ExternalEffectReliabilityDiagnosticEmitter(Protocol):
    """Pluggable sink for material ERL reliability observations."""

    def emit(self, observation: ExternalEffectReliabilityObservation) -> None:
        """Receive an observation; must not mutate ERL or execution state."""


class NullExternalEffectReliabilityDiagnosticEmitter:
    """Deterministic no-op emitter for composition defaults."""

    def emit(self, observation: ExternalEffectReliabilityObservation) -> None:
        return None


__all__ = [
    "ExternalEffectReliabilityDiagnosticEmitter",
    "NullExternalEffectReliabilityDiagnosticEmitter",
]
