# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Multi-analyzer coordination — ordering, isolation, aggregation only (W6-D)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from intergrax.contracts.runtime_intelligence.analyzer import (
    RuntimeIntelligenceAnalyzerOutcome,
    RuntimeIntelligenceAnalyzerPort,
    run_runtime_intelligence_analyzer_isolated,
)
from intergrax.contracts.runtime_intelligence.context import RuntimeIntelligenceContext


class RuntimeIntelligenceAnalyzerOrchestratorPort(Protocol):
    def orchestrate(
        self,
        context: RuntimeIntelligenceContext,
        analyzers: tuple[RuntimeIntelligenceAnalyzerPort, ...],
    ) -> RuntimeIntelligenceAnalyzerOrchestrationResult:
        ...


@dataclass(frozen=True, slots=True)
class RuntimeIntelligenceAnalyzerOrchestrationResult:
    """Aggregated per-analyzer outcomes for one immutable context snapshot."""

    context: RuntimeIntelligenceContext
    outcomes: tuple[RuntimeIntelligenceAnalyzerOutcome, ...]


@dataclass(frozen=True, slots=True)
class RuntimeIntelligenceAnalyzerOrchestrator:
    """
    Coordinate Runtime Intelligence analyzer execution.

    Does not build context, collect facts, decide policies, or execute recommendations.
    """

    def orchestrate(
        self,
        context: RuntimeIntelligenceContext,
        analyzers: tuple[RuntimeIntelligenceAnalyzerPort, ...],
    ) -> RuntimeIntelligenceAnalyzerOrchestrationResult:
        """
        Run analyzers in explicit caller order; empty collection yields no outcomes.

        Each analyzer is invoked through the W6-B isolated boundary.
        """
        outcomes = tuple(
            run_runtime_intelligence_analyzer_isolated(analyzer, context) for analyzer in analyzers
        )
        return RuntimeIntelligenceAnalyzerOrchestrationResult(context=context, outcomes=outcomes)


__all__ = [
    "RuntimeIntelligenceAnalyzerOrchestrationResult",
    "RuntimeIntelligenceAnalyzerOrchestrator",
    "RuntimeIntelligenceAnalyzerOrchestratorPort",
]
