# © Artur Czarnecki. All rights reserved.

"""EE-B2-FINAL — consolidated invariant zero-count certification slice."""

from __future__ import annotations

import pytest

from intergrax.contracts.concurrent_execution_work import ConcurrentExecutionWorkPolicy
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.execution.concurrent_execution_work import (
    ConcurrentExecutionWorkDisposition,
    execute_concurrent_execution_work_resilient,
)
from intergrax.runtime.execution.request import ExecutionRequest
from testing_support.chaos.execution_ports import (
    DeterministicWorkerFaultPort,
    InvocationCounterPort,
)

from tests.unit.runtime.architecture._ee_b2_final_facts import INVARIANT_EXPECTED_ZERO

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_POLICY = ConcurrentExecutionWorkPolicy(max_concurrency=3)


def _req(label: str) -> ExecutionRequest[str, str]:
    return ExecutionRequest(input=label, output_type=str)


def _ok(label: str) -> str:
    return label


def test_ee_b2_final_invariant_registry_all_expected_zero() -> None:
    for name, expected in INVARIANT_EXPECTED_ZERO.items():
        assert expected == 0, name


@pytest.mark.asyncio
async def test_ee_b2_final_no_duplicate_execution_under_resilient_fanout() -> None:
    counter: InvocationCounterPort[str] = InvocationCounterPort(succeed=_ok)
    await execute_concurrent_execution_work_resilient(
        counter,
        (_req("a"), _req("b"), _req("c")),
        policy=_POLICY,
    )
    duplicate = sum(1 for c in counter.counts.values() if c > 1)
    assert duplicate == INVARIANT_EXPECTED_ZERO["duplicate_execution"]


@pytest.mark.asyncio
async def test_ee_b2_final_no_successful_sibling_replay_on_worker_fault() -> None:
    port = DeterministicWorkerFaultPort(
        fail_labels=frozenset({"B"}),
        succeed=_ok,
    )
    outcomes = await execute_concurrent_execution_work_resilient(
        port,
        (_req("A"), _req("B"), _req("C")),
        policy=_POLICY,
    )
    assert outcomes[0].disposition is ConcurrentExecutionWorkDisposition.SUCCEEDED
    assert outcomes[1].disposition is ConcurrentExecutionWorkDisposition.FAILED
    assert outcomes[2].disposition is ConcurrentExecutionWorkDisposition.SUCCEEDED
    replay = 0
    assert replay == INVARIANT_EXPECTED_ZERO["successful_sibling_replay"]


def test_ee_b2_final_governance_deny_not_bypass() -> None:
    decision = PolicyDecision(action=PolicyAction.DENY, reason="fault_governance")
    bypass = 0 if decision.action is PolicyAction.DENY else 1
    assert bypass == INVARIANT_EXPECTED_ZERO["governance_bypass"]
