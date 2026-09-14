# © Artur Czarnecki. All rights reserved.

"""EE-B1.1 — worker failure isolation (Scenario A/B semantics)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from intergrax.contracts.concurrent_execution_work import ConcurrentExecutionWorkPolicy
from intergrax.contracts.execution_reliability import (
    ExecutionFailureContext,
    ExecutionFailureSemanticCategory,
)
from intergrax.contracts.resilience_policy import FailureClass
from intergrax.runtime.execution.concurrent_execution_work import (
    ConcurrentExecutionWorkDisposition,
    execute_concurrent_execution_work_resilient,
)
from intergrax.runtime.execution.reliability import default_execution_failure_classifier
from intergrax.runtime.execution.execution_work_port import ExecutionWorkPort
from intergrax.runtime.execution.request import ExecutionRequest

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@dataclass(frozen=True, slots=True)
class _WorkResult:
    value: str


def _request(label: str) -> ExecutionRequest[str, _WorkResult]:
    return ExecutionRequest(input=label, output_type=_WorkResult)


class _FlakyPort(ExecutionWorkPort[str, _WorkResult, _WorkResult]):
    def __init__(self, fail_label: str) -> None:
        self._fail_label = fail_label

    async def execute(self, request: ExecutionRequest[str, _WorkResult]) -> _WorkResult:
        if request.input == self._fail_label:
            raise RuntimeError("worker_a_failed")
        await asyncio.sleep(0)
        return _WorkResult(value=request.input)


@pytest.mark.asyncio
async def test_ee_b1_1_scenario_a_worker_failure_does_not_abort_siblings() -> None:
    requests = (_request("0"), _request("1"), _request("2"))
    outcomes = await execute_concurrent_execution_work_resilient(
        _FlakyPort(fail_label="0"),
        requests,
        policy=ConcurrentExecutionWorkPolicy(max_concurrency=3),
    )
    assert len(outcomes) == 3
    assert outcomes[0].disposition is ConcurrentExecutionWorkDisposition.FAILED
    assert outcomes[1].disposition is ConcurrentExecutionWorkDisposition.SUCCEEDED
    assert outcomes[2].disposition is ConcurrentExecutionWorkDisposition.SUCCEEDED


def test_ee_b1_1_scenario_b_dependency_failure_classified_without_provider_coupling() -> None:
    classifier = default_execution_failure_classifier()
    decision = classifier.classify(
        ExecutionFailureContext(
            failure_class=FailureClass.DEPENDENCY_ERROR,
            dependency_unavailable=True,
            reason="upstream_unavailable",
        ),
    )
    assert decision.category is ExecutionFailureSemanticCategory.DEPENDENCY_FAILURE
    assert decision.retry_projection is not None
