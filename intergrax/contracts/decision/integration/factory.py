# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Factory for assembling the integration engine from injected composition only."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.decision.integration.composition import (
    DecisionIntegrationCompositionProvider,
)
from intergrax.contracts.decision.integration.engine import (
    DecisionSystemIntegrationEngine,
)


@dataclass(frozen=True, slots=True)
class DecisionSystemIntegrationFactory:
    """Builds ``DecisionSystemIntegrationEngine`` — knows composition, not concrete adapters."""

    @staticmethod
    def create_engine(
        composition: DecisionIntegrationCompositionProvider,
    ) -> DecisionSystemIntegrationEngine:
        if not isinstance(composition, DecisionIntegrationCompositionProvider):
            raise TypeError(
                "composition must implement DecisionIntegrationCompositionProvider",
            )
        return DecisionSystemIntegrationEngine(
            adapter_providers=composition.adapter_providers,
            audit_provider=composition.audit_provider,
        )


__all__ = ["DecisionSystemIntegrationFactory"]
