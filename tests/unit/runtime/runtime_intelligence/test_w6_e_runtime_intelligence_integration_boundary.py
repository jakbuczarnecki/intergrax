# © Artur Czarnecki. All rights reserved.

"""W6-E execution runtime integration boundary — contracts, isolation, ownership."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.contracts.runtime_intelligence import (
    ANALYZER_OUTCOME_OK,
    ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE,
    INTEGRATION_OUTCOME_INVALID_INPUT,
    INTEGRATION_OUTCOME_OK,
    INTEGRATION_OUTCOME_UNAVAILABLE,
    AnalyzerExecutionError,
    RuntimeIntelligenceContext,
    RuntimeIntelligenceFactKind,
    RuntimeIntelligenceFactReference,
    RuntimeIntelligenceFactsInput,
    RuntimeIntelligenceResult,
    RuntimeIntelligenceRuntimeIntegrationPort,
    invoke_runtime_intelligence_integration_isolated,
)
from intergrax.contracts.runtime_intelligence.errors import RuntimeIntelligenceError
from intergrax.runtime.execution.runtime_intelligence_advisory import (
    request_execution_runtime_intelligence_advisory,
)
from intergrax.runtime.runtime_intelligence import (
    DeterministicRuntimeIntelligenceAnalyzer,
    RuntimeIntelligenceFacts,
    RuntimeIntelligenceService,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_COLLECTED_AT = datetime(2026, 9, 12, 12, 0, tzinfo=UTC)


def _fact_ref() -> RuntimeIntelligenceFactReference:
    return RuntimeIntelligenceFactReference(
        fact_kind=RuntimeIntelligenceFactKind.RUNTIME_EVENT,
        fact_ref="evt_00000000000000000000000000000001",
    )


def _facts_input() -> RuntimeIntelligenceFactsInput:
    return RuntimeIntelligenceFactsInput(
        tenant_id="tenant-a",
        task_id="task_00000000000000000000000000000001",
        run_id="run_00000000000000000000000000000001",
        attempt_id="attempt_00000000000000000000000000000001",
        fact_references=(_fact_ref(),),
        correlation_id="corr-w6e-boundary",
        collected_at=_COLLECTED_AT,
    )


def _runtime_facts() -> RuntimeIntelligenceFacts:
    return RuntimeIntelligenceFacts(
        tenant_id="tenant-a",
        task_id="task_00000000000000000000000000000001",
        run_id="run_00000000000000000000000000000001",
        attempt_id="attempt_00000000000000000000000000000001",
        fact_references=(_fact_ref(),),
        correlation_id="corr-w6e-boundary",
        collected_at=_COLLECTED_AT,
    )


class _ExplodingPort:
    def analyze_advisory(
        self,
        facts: RuntimeIntelligenceFactsInput,
    ) -> object:
        raise RuntimeIntelligenceError("simulated service failure")


class _ExplodingAnalyzer:
    analyzer_id = "plugin.exploding"
    analyzer_version = "1.0.0"

    def analyze(self, context: RuntimeIntelligenceContext) -> RuntimeIntelligenceResult:
        raise AnalyzerExecutionError("simulated analyzer failure")


def test_execution_runtime_can_invoke_intelligence_and_receive_advisory() -> None:
    service: RuntimeIntelligenceRuntimeIntegrationPort = RuntimeIntelligenceService()
    outcome = request_execution_runtime_intelligence_advisory(service, _runtime_facts())
    assert outcome is not None
    assert outcome.outcome == INTEGRATION_OUTCOME_OK
    assert outcome.advisory is not None
    assert outcome.advisory.outcomes[0].outcome == ANALYZER_OUTCOME_OK
    assert outcome.advisory.outcomes[0].result is not None


def test_unwired_port_returns_none_without_touching_execution() -> None:
    assert request_execution_runtime_intelligence_advisory(None, _runtime_facts()) is None


def test_integration_failure_isolation_does_not_raise_to_caller() -> None:
    outcome = invoke_runtime_intelligence_integration_isolated(
        _ExplodingPort(),
        _facts_input(),
    )
    assert outcome.outcome == INTEGRATION_OUTCOME_UNAVAILABLE
    assert outcome.advisory is None


def test_analyzer_failure_isolated_within_advisory_response() -> None:
    service = RuntimeIntelligenceService(analyzers=(_ExplodingAnalyzer(),))
    outcome = invoke_runtime_intelligence_integration_isolated(service, _facts_input())
    assert outcome.outcome == INTEGRATION_OUTCOME_OK
    assert outcome.advisory is not None
    assert outcome.advisory.outcomes[0].outcome == ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE


def test_intelligence_port_has_no_execution_authority_surface() -> None:
    port_methods = {
        name
        for name in dir(RuntimeIntelligenceRuntimeIntegrationPort)
        if not name.startswith("_")
    }
    assert port_methods == {"analyze_advisory"}


async def _execution_hot_path_stub(
    port: RuntimeIntelligenceRuntimeIntegrationPort | None,
) -> str:
    request_execution_runtime_intelligence_advisory(port, _runtime_facts())
    return "executed"


@pytest.mark.asyncio
async def test_execution_continues_when_intelligence_fails() -> None:
    result = await _execution_hot_path_stub(_ExplodingPort())
    assert result == "executed"


def test_request_scoped_lifecycle_produces_independent_advisory() -> None:
    service = RuntimeIntelligenceService()
    first = invoke_runtime_intelligence_integration_isolated(service, _facts_input())
    second_input = RuntimeIntelligenceFactsInput(
        tenant_id="tenant-a",
        task_id="task_00000000000000000000000000000001",
        run_id="run_00000000000000000000000000000001",
        attempt_id="attempt_00000000000000000000000000000001",
        fact_references=(_fact_ref(),),
        correlation_id="corr-w6e-other",
        collected_at=_COLLECTED_AT,
    )
    second = invoke_runtime_intelligence_integration_isolated(service, second_input)
    assert first.advisory is not None and second.advisory is not None
    assert (
        first.advisory.context.metadata.correlation_id
        != second.advisory.context.metadata.correlation_id
    )


def test_invalid_facts_input_yields_invalid_outcome_not_exception() -> None:
    invalid = RuntimeIntelligenceFactsInput(
        tenant_id="tenant-a",
        task_id="task_00000000000000000000000000000001",
        run_id="run_00000000000000000000000000000001",
        fact_references=(),
        correlation_id="corr-invalid",
        collected_at=_COLLECTED_AT,
    )
    outcome = invoke_runtime_intelligence_integration_isolated(
        RuntimeIntelligenceService(),
        invalid,
    )
    assert outcome.outcome == INTEGRATION_OUTCOME_INVALID_INPUT


def test_deterministic_analyzer_plugin_compatible_through_service() -> None:
    service = RuntimeIntelligenceService(
        analyzers=(DeterministicRuntimeIntelligenceAnalyzer(),)
    )
    outcome = invoke_runtime_intelligence_integration_isolated(service, _facts_input())
    assert outcome.advisory is not None
    result = outcome.advisory.outcomes[0].result
    assert result is not None
    assert result.analyzer_id == "runtime_intelligence.deterministic"
