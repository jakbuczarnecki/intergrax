# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Request-scoped intelligence analyze lifecycle — no background workers (W6-C)."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.runtime_intelligence.analyzer import (
    RuntimeIntelligenceAnalyzerOutcome,
    RuntimeIntelligenceAnalyzerPort,
    run_runtime_intelligence_analyzer_isolated,
)
from intergrax.contracts.runtime_intelligence.context import RuntimeIntelligenceContext
from intergrax.runtime.runtime_intelligence.context_builder import RuntimeIntelligenceContextBuilder
from intergrax.runtime.runtime_intelligence.runtime_facts import RuntimeIntelligenceFacts


@dataclass(frozen=True, slots=True)
class RuntimeIntelligenceAnalysisRequest:
    """Owns facts + analyzer reference for one advisory invocation."""

    facts: RuntimeIntelligenceFacts
    analyzer: RuntimeIntelligenceAnalyzerPort
    context_builder: RuntimeIntelligenceContextBuilder = field(
        default_factory=RuntimeIntelligenceContextBuilder
    )


@dataclass(frozen=True, slots=True)
class RuntimeIntelligenceAnalysisResponse:
    context: RuntimeIntelligenceContext
    outcome: RuntimeIntelligenceAnalyzerOutcome


def run_runtime_intelligence_analysis(
    request: RuntimeIntelligenceAnalysisRequest,
) -> RuntimeIntelligenceAnalysisResponse:
    """
    create request → build context → analyze → return result.

    Does not mutate execution state; caller releases request-scoped resources.
    """
    context = request.context_builder.build(request.facts)
    outcome = run_runtime_intelligence_analyzer_isolated(request.analyzer, context)
    return RuntimeIntelligenceAnalysisResponse(context=context, outcome=outcome)


__all__ = [
    "RuntimeIntelligenceAnalysisRequest",
    "RuntimeIntelligenceAnalysisResponse",
    "run_runtime_intelligence_analysis",
]
