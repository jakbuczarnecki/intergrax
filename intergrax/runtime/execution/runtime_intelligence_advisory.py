# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Optional execution-runtime advisory intelligence hook — isolated, non-authoritative (W6-E)."""

from __future__ import annotations

from intergrax.contracts.runtime_intelligence.integration import (
    RuntimeIntelligenceFactsInput,
    RuntimeIntelligenceIntegrationOutcome,
    RuntimeIntelligenceRuntimeIntegrationPort,
    invoke_runtime_intelligence_integration_isolated,
)
from intergrax.runtime.runtime_intelligence.runtime_facts import RuntimeIntelligenceFacts


def runtime_intelligence_facts_to_input(
    facts: RuntimeIntelligenceFacts,
) -> RuntimeIntelligenceFactsInput:
    return RuntimeIntelligenceFactsInput(
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


def request_execution_runtime_intelligence_advisory(
    port: RuntimeIntelligenceRuntimeIntegrationPort | None,
    facts: RuntimeIntelligenceFacts,
) -> RuntimeIntelligenceIntegrationOutcome | None:
    """
    Execution-owned call site: optional port, fail-soft, no lifecycle side effects.

    Returns None when intelligence is not wired; otherwise an integration outcome.
    """
    if port is None:
        return None
    return invoke_runtime_intelligence_integration_isolated(
        port,
        runtime_intelligence_facts_to_input(facts),
    )


__all__ = [
    "request_execution_runtime_intelligence_advisory",
    "runtime_intelligence_facts_to_input",
]
