# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Execution-runtime integration port — advisory only, fail-soft (W6-E)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from intergrax.contracts.runtime_intelligence.analyzer import RuntimeIntelligenceAnalyzerOutcome
from intergrax.contracts.runtime_intelligence.context import (
    RuntimeIntelligenceContext,
    RuntimeIntelligenceFactReference,
)
from intergrax.contracts.runtime_intelligence.errors import (
    InvalidIntelligenceContextError,
    RuntimeIntelligenceError,
)

INTEGRATION_OUTCOME_OK = "ok"
INTEGRATION_OUTCOME_INVALID_INPUT = "INVALID_INPUT"
INTEGRATION_OUTCOME_UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class RuntimeIntelligenceFactsInput:
    """
    Contract-level read-only fact snapshot supplied by Execution Runtime.

    Ownership: execution caller assembles pointers; intelligence never mutates stores.
    """

    tenant_id: str
    task_id: str
    run_id: str
    fact_references: tuple[RuntimeIntelligenceFactReference, ...]
    correlation_id: str
    collected_at: datetime
    attempt_id: str | None = None
    execution_id: str | None = None
    context_label: str = ""

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty")
        if not self.correlation_id.strip():
            raise ValueError("correlation_id must be non-empty")
        if self.collected_at.tzinfo is None:
            raise ValueError("collected_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class RuntimeIntelligenceAdvisoryResponse:
    """Advisory envelope returned to execution — no execution hooks."""

    context: RuntimeIntelligenceContext
    outcomes: tuple[RuntimeIntelligenceAnalyzerOutcome, ...]


@dataclass(frozen=True, slots=True)
class RuntimeIntelligenceIntegrationOutcome:
    """Fail-soft integration boundary record — not execution truth."""

    outcome: str
    advisory: RuntimeIntelligenceAdvisoryResponse | None = None


@runtime_checkable
class RuntimeIntelligenceRuntimeIntegrationPort(Protocol):
    """Dependency-inverted boundary from Execution Runtime to Runtime Intelligence."""

    def analyze_advisory(
        self,
        facts: RuntimeIntelligenceFactsInput,
    ) -> RuntimeIntelligenceAdvisoryResponse:
        """Request-scoped advisory analysis — must not mutate execution state."""
        ...


def invoke_runtime_intelligence_integration_isolated(
    port: RuntimeIntelligenceRuntimeIntegrationPort,
    facts: RuntimeIntelligenceFactsInput,
) -> RuntimeIntelligenceIntegrationOutcome:
    """
    Fail-soft execution hot-path boundary.

    Intelligence errors never propagate as generic execution exceptions.
    """
    try:
        advisory = port.analyze_advisory(facts)
    except InvalidIntelligenceContextError:
        return RuntimeIntelligenceIntegrationOutcome(
            outcome=INTEGRATION_OUTCOME_INVALID_INPUT,
            advisory=None,
        )
    except RuntimeIntelligenceError:
        return RuntimeIntelligenceIntegrationOutcome(
            outcome=INTEGRATION_OUTCOME_UNAVAILABLE,
            advisory=None,
        )
    except ValueError:
        return RuntimeIntelligenceIntegrationOutcome(
            outcome=INTEGRATION_OUTCOME_INVALID_INPUT,
            advisory=None,
        )
    return RuntimeIntelligenceIntegrationOutcome(
        outcome=INTEGRATION_OUTCOME_OK,
        advisory=advisory,
    )


__all__ = [
    "INTEGRATION_OUTCOME_INVALID_INPUT",
    "INTEGRATION_OUTCOME_OK",
    "INTEGRATION_OUTCOME_UNAVAILABLE",
    "RuntimeIntelligenceAdvisoryResponse",
    "RuntimeIntelligenceFactsInput",
    "RuntimeIntelligenceIntegrationOutcome",
    "RuntimeIntelligenceRuntimeIntegrationPort",
    "invoke_runtime_intelligence_integration_isolated",
]
