# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Enterprise Runtime Intelligence integration layer (W6-C + W6-E facade)."""

from intergrax.runtime.runtime_intelligence.analysis_request import (
    RuntimeIntelligenceAnalysisRequest,
    RuntimeIntelligenceAnalysisResponse,
    RuntimeIntelligenceOrchestratedAnalysisRequest,
    RuntimeIntelligenceOrchestratedAnalysisResponse,
    run_runtime_intelligence_analysis,
    run_runtime_intelligence_orchestrated_analysis,
)
from intergrax.runtime.runtime_intelligence.analyzer_orchestrator import (
    RuntimeIntelligenceAnalyzerOrchestrationResult,
    RuntimeIntelligenceAnalyzerOrchestrator,
    RuntimeIntelligenceAnalyzerOrchestratorPort,
)
from intergrax.runtime.runtime_intelligence.context_builder import (
    RuntimeIntelligenceContextBuilder,
    intelligence_signal_from_fact_ref,
)
from intergrax.runtime.runtime_intelligence.deterministic_analyzer import (
    DeterministicRuntimeIntelligenceAnalyzer,
)
from intergrax.runtime.runtime_intelligence.facade import RuntimeIntelligenceFacade
from intergrax.runtime.runtime_intelligence.service import RuntimeIntelligenceService
from intergrax.runtime.runtime_intelligence.runtime_facts import (
    RuntimeIntelligenceFacts,
    RuntimeIntelligenceObservedSignal,
    RuntimeIntelligenceSignalKind,
)

__all__ = [
    "DeterministicRuntimeIntelligenceAnalyzer",
    "RuntimeIntelligenceAnalysisRequest",
    "RuntimeIntelligenceAnalysisResponse",
    "RuntimeIntelligenceAnalyzerOrchestrationResult",
    "RuntimeIntelligenceAnalyzerOrchestrator",
    "RuntimeIntelligenceAnalyzerOrchestratorPort",
    "RuntimeIntelligenceOrchestratedAnalysisRequest",
    "RuntimeIntelligenceOrchestratedAnalysisResponse",
    "RuntimeIntelligenceContextBuilder",
    "RuntimeIntelligenceFacade",
    "RuntimeIntelligenceService",
    "RuntimeIntelligenceFacts",
    "RuntimeIntelligenceObservedSignal",
    "RuntimeIntelligenceSignalKind",
    "intelligence_signal_from_fact_ref",
    "run_runtime_intelligence_analysis",
    "run_runtime_intelligence_orchestrated_analysis",
]
