# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Enterprise Runtime Intelligence contracts (W6-B) — advisory plane SPI."""

from intergrax.contracts.runtime_intelligence.analyzer import (
    ANALYZER_OUTCOME_INVALID_CONTEXT,
    ANALYZER_OUTCOME_OK,
    ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE,
    RuntimeIntelligenceAnalyzerOutcome,
    RuntimeIntelligenceAnalyzerPort,
    run_runtime_intelligence_analyzer_isolated,
)
from intergrax.contracts.runtime_intelligence.context import (
    RUNTIME_INTELLIGENCE_CONTEXT_SCHEMA_VERSION,
    RuntimeIntelligenceContext,
    RuntimeIntelligenceContextMetadata,
    RuntimeIntelligenceFactKind,
    RuntimeIntelligenceFactReference,
    validate_runtime_intelligence_context,
)
from intergrax.contracts.runtime_intelligence.integration import (
    INTEGRATION_OUTCOME_INVALID_INPUT,
    INTEGRATION_OUTCOME_OK,
    INTEGRATION_OUTCOME_UNAVAILABLE,
    RuntimeIntelligenceAdvisoryResponse,
    RuntimeIntelligenceFactsInput,
    RuntimeIntelligenceIntegrationOutcome,
    RuntimeIntelligenceRuntimeIntegrationPort,
    invoke_runtime_intelligence_integration_isolated,
)
from intergrax.contracts.runtime_intelligence.errors import (
    AnalyzerExecutionError,
    InvalidIntelligenceContextError,
    RuntimeIntelligenceError,
)
from intergrax.contracts.runtime_intelligence.evidence import (
    IntelligenceEvidence,
    IntelligenceEvidenceSourceKind,
)
from intergrax.contracts.runtime_intelligence.recommendation import (
    IntelligenceRecommendation,
    IntelligenceRecommendationKind,
)
from intergrax.contracts.runtime_intelligence.result import (
    RUNTIME_INTELLIGENCE_RESULT_SCHEMA_VERSION,
    RuntimeIntelligenceResult,
)

__all__ = [
    "INTEGRATION_OUTCOME_INVALID_INPUT",
    "INTEGRATION_OUTCOME_OK",
    "INTEGRATION_OUTCOME_UNAVAILABLE",
    "ANALYZER_OUTCOME_INVALID_CONTEXT",
    "ANALYZER_OUTCOME_OK",
    "ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE",
    "AnalyzerExecutionError",
    "IntelligenceEvidence",
    "IntelligenceEvidenceSourceKind",
    "IntelligenceRecommendation",
    "IntelligenceRecommendationKind",
    "InvalidIntelligenceContextError",
    "RuntimeIntelligenceAdvisoryResponse",
    "RuntimeIntelligenceFactsInput",
    "RuntimeIntelligenceIntegrationOutcome",
    "RuntimeIntelligenceRuntimeIntegrationPort",
    "invoke_runtime_intelligence_integration_isolated",
    "RUNTIME_INTELLIGENCE_CONTEXT_SCHEMA_VERSION",
    "RUNTIME_INTELLIGENCE_RESULT_SCHEMA_VERSION",
    "RuntimeIntelligenceAnalyzerOutcome",
    "RuntimeIntelligenceAnalyzerPort",
    "RuntimeIntelligenceContext",
    "RuntimeIntelligenceContextMetadata",
    "RuntimeIntelligenceError",
    "RuntimeIntelligenceFactKind",
    "RuntimeIntelligenceFactReference",
    "RuntimeIntelligenceResult",
    "run_runtime_intelligence_analyzer_isolated",
    "validate_runtime_intelligence_context",
]
