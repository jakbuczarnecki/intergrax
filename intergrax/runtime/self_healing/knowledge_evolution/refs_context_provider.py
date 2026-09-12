# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Derive minimal operating context from evolution scope refs (R5.5)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.knowledge_evolution.contextual.context_provider import (
    StrategyContextResolutionRequest,
)
from intergrax.contracts.self_healing.knowledge_evolution.contextual.operating_context import (
    StrategyKnowledgeOperatingContext,
)


@dataclass(frozen=True, slots=True)
class EvolutionRefsContextProvider:
    provider_id: str = "platform.evolution_refs_context"

    def resolve(
        self,
        request: StrategyContextResolutionRequest,
    ) -> StrategyKnowledgeOperatingContext | None:
        scope = request.evolution_scope
        if scope.diagnostic_investigation_id is not None:
            return StrategyKnowledgeOperatingContext(
                problem_source=f"investigation:{scope.diagnostic_investigation_id}",
                descriptor_refs=scope.context_refs,
                provider_id=self.provider_id,
            )
        if scope.context_refs:
            return StrategyKnowledgeOperatingContext(
                descriptor_refs=scope.context_refs,
                provider_id=self.provider_id,
            )
        return None


__all__ = ["EvolutionRefsContextProvider"]
