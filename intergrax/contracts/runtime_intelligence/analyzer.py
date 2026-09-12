# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Analyzer plugin SPI — dependency-inverted, framework-agnostic (W6-B)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.contracts.runtime_intelligence.context import (
    RuntimeIntelligenceContext,
    validate_runtime_intelligence_context,
)
from intergrax.contracts.runtime_intelligence.errors import (
    AnalyzerExecutionError,
    InvalidIntelligenceContextError,
    RuntimeIntelligenceError,
)
from intergrax.contracts.runtime_intelligence.result import RuntimeIntelligenceResult

ANALYZER_OUTCOME_OK = "ok"
ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE = "PLUGIN_UNAVAILABLE"
ANALYZER_OUTCOME_INVALID_CONTEXT = "INVALID_CONTEXT"


@runtime_checkable
class RuntimeIntelligenceAnalyzerPort(Protocol):
    """Replaceable analyzer — local, ML, or external service adapter."""

    @property
    def analyzer_id(self) -> str: ...

    @property
    def analyzer_version(self) -> str: ...

    def analyze(self, context: RuntimeIntelligenceContext) -> RuntimeIntelligenceResult:
        """Pure analysis over immutable context — no execution authority."""
        ...


@dataclass(frozen=True, slots=True)
class RuntimeIntelligenceAnalyzerOutcome:
    """Per-analyzer invocation record for orchestration audit — not execution truth."""

    analyzer_id: str
    outcome: str
    result: RuntimeIntelligenceResult | None = None


def run_runtime_intelligence_analyzer_isolated(
    analyzer: RuntimeIntelligenceAnalyzerPort,
    context: RuntimeIntelligenceContext,
) -> RuntimeIntelligenceAnalyzerOutcome:
    """
    Fail-soft single-analyzer boundary.

    Analyzer failure yields PLUGIN_UNAVAILABLE; execution owners remain authoritative.
    """
    try:
        validate_runtime_intelligence_context(context)
    except InvalidIntelligenceContextError:
        return RuntimeIntelligenceAnalyzerOutcome(
            analyzer_id=analyzer.analyzer_id,
            outcome=ANALYZER_OUTCOME_INVALID_CONTEXT,
            result=None,
        )
    try:
        result = analyzer.analyze(context)
    except AnalyzerExecutionError:
        return RuntimeIntelligenceAnalyzerOutcome(
            analyzer_id=analyzer.analyzer_id,
            outcome=ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE,
            result=None,
        )
    except RuntimeIntelligenceError:
        return RuntimeIntelligenceAnalyzerOutcome(
            analyzer_id=analyzer.analyzer_id,
            outcome=ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE,
            result=None,
        )
    if result.analyzer_id != analyzer.analyzer_id:
        return RuntimeIntelligenceAnalyzerOutcome(
            analyzer_id=analyzer.analyzer_id,
            outcome=ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE,
            result=None,
        )
    return RuntimeIntelligenceAnalyzerOutcome(
        analyzer_id=analyzer.analyzer_id,
        outcome=ANALYZER_OUTCOME_OK,
        result=result,
    )


__all__ = [
    "ANALYZER_OUTCOME_INVALID_CONTEXT",
    "ANALYZER_OUTCOME_OK",
    "ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE",
    "RuntimeIntelligenceAnalyzerOutcome",
    "RuntimeIntelligenceAnalyzerPort",
    "run_runtime_intelligence_analyzer_isolated",
]
