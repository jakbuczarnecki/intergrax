# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Application-facing Runtime Intelligence boundary — lifecycle coordination only (W6-E)."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.runtime_intelligence.analyzer import RuntimeIntelligenceAnalyzerPort
from intergrax.runtime.runtime_intelligence.analysis_request import (
    RuntimeIntelligenceAnalysisRequest,
    RuntimeIntelligenceAnalysisResponse,
    RuntimeIntelligenceOrchestratedAnalysisRequest,
    RuntimeIntelligenceOrchestratedAnalysisResponse,
    run_runtime_intelligence_analysis,
    run_runtime_intelligence_orchestrated_analysis,
)
from intergrax.runtime.runtime_intelligence.analyzer_orchestrator import (
    RuntimeIntelligenceAnalyzerOrchestrator,
)
from intergrax.runtime.runtime_intelligence.context_builder import RuntimeIntelligenceContextBuilder
from intergrax.runtime.runtime_intelligence.runtime_facts import RuntimeIntelligenceFacts


@dataclass(frozen=True, slots=True)
class RuntimeIntelligenceFacade:
    """
    Stable entry point for advisory Runtime Intelligence analysis.

    Coordinates context projection and analyzer execution; does not execute
    recommendations, mutate runtime state, or own policies.
    """

    context_builder: RuntimeIntelligenceContextBuilder = field(
        default_factory=RuntimeIntelligenceContextBuilder
    )
    orchestrator: RuntimeIntelligenceAnalyzerOrchestrator = field(
        default_factory=RuntimeIntelligenceAnalyzerOrchestrator
    )

    def analyze(
        self,
        facts: RuntimeIntelligenceFacts,
        analyzer: RuntimeIntelligenceAnalyzerPort,
    ) -> RuntimeIntelligenceAnalysisResponse:
        """Single-analyzer advisory path: build context → isolated analyze → response."""
        return run_runtime_intelligence_analysis(
            RuntimeIntelligenceAnalysisRequest(
                facts=facts,
                analyzer=analyzer,
                context_builder=self.context_builder,
            )
        )

    def analyze_orchestrated(
        self,
        facts: RuntimeIntelligenceFacts,
        analyzers: tuple[RuntimeIntelligenceAnalyzerPort, ...],
    ) -> RuntimeIntelligenceOrchestratedAnalysisResponse:
        """Multi-analyzer advisory path: build context → orchestrate → response."""
        return run_runtime_intelligence_orchestrated_analysis(
            RuntimeIntelligenceOrchestratedAnalysisRequest(
                facts=facts,
                analyzers=analyzers,
                context_builder=self.context_builder,
                orchestrator=self.orchestrator,
            )
        )


__all__ = ["RuntimeIntelligenceFacade"]
