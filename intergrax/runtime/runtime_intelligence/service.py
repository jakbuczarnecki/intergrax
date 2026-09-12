# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Runtime Intelligence service — implements execution integration port (W6-E)."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.runtime_intelligence.analyzer import RuntimeIntelligenceAnalyzerPort
from intergrax.contracts.runtime_intelligence.integration import (
    RuntimeIntelligenceAdvisoryResponse,
    RuntimeIntelligenceFactsInput,
    RuntimeIntelligenceRuntimeIntegrationPort,
)
from intergrax.runtime.runtime_intelligence.deterministic_analyzer import (
    DeterministicRuntimeIntelligenceAnalyzer,
)
from intergrax.runtime.runtime_intelligence.facade import RuntimeIntelligenceFacade
from intergrax.runtime.runtime_intelligence.runtime_facts import RuntimeIntelligenceFacts


def _facts_from_input(facts: RuntimeIntelligenceFactsInput) -> RuntimeIntelligenceFacts:
    return RuntimeIntelligenceFacts(
        tenant_id=facts.tenant_id,
        task_id=facts.task_id,
        run_id=facts.run_id,
        attempt_id=facts.attempt_id,
        execution_id=facts.execution_id,
        fact_references=facts.fact_references,
        correlation_id=facts.correlation_id,
        collected_at=facts.collected_at,
        context_label=facts.context_label,
    )


@dataclass(frozen=True, slots=True)
class RuntimeIntelligenceService(RuntimeIntelligenceRuntimeIntegrationPort):
    """
    Advisory intelligence owner for one composition root.

    Does not own execution lifecycle, policies, or recommendation execution.
    """

    analyzers: tuple[RuntimeIntelligenceAnalyzerPort, ...] = field(
        default_factory=lambda: (DeterministicRuntimeIntelligenceAnalyzer(),)
    )
    facade: RuntimeIntelligenceFacade = field(default_factory=RuntimeIntelligenceFacade)

    def analyze_advisory(
        self,
        facts: RuntimeIntelligenceFactsInput,
    ) -> RuntimeIntelligenceAdvisoryResponse:
        runtime_facts = _facts_from_input(facts)
        if len(self.analyzers) == 1:
            response = self.facade.analyze(runtime_facts, self.analyzers[0])
            return RuntimeIntelligenceAdvisoryResponse(
                context=response.context,
                outcomes=(response.outcome,),
            )
        orchestrated = self.facade.analyze_orchestrated(runtime_facts, self.analyzers)
        return RuntimeIntelligenceAdvisoryResponse(
            context=orchestrated.context,
            outcomes=orchestrated.orchestration.outcomes,
        )


__all__ = ["RuntimeIntelligenceService"]
