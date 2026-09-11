# © Artur Czarnecki. All rights reserved.

"""P0 — enterprise execution scale & resilience static inventory gates."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from intergrax.agent_distribution.bounded_multi_agent_fanout import (
    MAX_FAN_OUT_CONCURRENCY,
    MAX_FAN_OUT_ITEMS,
)
from intergrax.runtime.execution.retry import classify_execution_failure
from intergrax.contracts.execution_retry import (
    ExecutionFailureKind,
    ExecutionRetryAction,
    ExecutionRetryEligibilityRequest,
)
from intergrax.contracts.orchestration_topology import (
    OrchestrationSchedulingPolicy,
    resolve_effective_orchestration_concurrency,
)
from intergrax.runtime.execution.retry.policy import evaluate_execution_retry_eligibility
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FANOUT_MODULE = (
    _REPO_ROOT / "intergrax" / "agent_distribution" / "bounded_multi_agent_fanout.py"
)
_GRAPH_RUNNER = (
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "orchestration" / "graph_runner.py"
)


def test_fan_out_platform_bounds_frozen() -> None:
    assert MAX_FAN_OUT_ITEMS == 256
    assert MAX_FAN_OUT_CONCURRENCY == 64


def test_fan_out_validation_enforces_platform_limits_in_source() -> None:
    source = _FANOUT_MODULE.read_text(encoding="utf-8")
    assert "len(request.items) > MAX_FAN_OUT_ITEMS" in source
    assert "request.max_concurrency > MAX_FAN_OUT_CONCURRENCY" in source


def test_orchestration_concurrency_unbounded_when_limits_absent() -> None:
    assert resolve_effective_orchestration_concurrency(None, None) is None
    assert resolve_effective_orchestration_concurrency(10, 3) == 3


def test_r1_global_deadline_blocks_retry_when_backoff_exceeds() -> None:
    result = evaluate_execution_retry_eligibility(
        ExecutionRetryEligibilityRequest(
            classification=classify_execution_failure(
                kind=ExecutionFailureKind.RETRYABLE_TRANSIENT,
                reason="inventory",
            ),
            attempt_number=1,
            max_attempts=5,
            proposed_backoff_seconds=10.0,
            now_monotonic=995.0,
            global_deadline_monotonic=1000.0,
        )
    )
    assert result.action is ExecutionRetryAction.FAIL
    assert result.reason == "global_deadline_exceeded"


def test_graph_executor_exposes_optional_parallel_caps() -> None:
    params = inspect.signature(GraphExecutor.__init__).parameters
    assert "max_parallel_nodes" in params
    assert "max_inflight_nodes" in params


def test_concurrent_execution_work_policy_contract_explicit_required() -> None:
    from intergrax.contracts.concurrent_execution_work import ConcurrentExecutionWorkPolicy

    policy = ConcurrentExecutionWorkPolicy(max_concurrency=8)
    assert policy.max_concurrency == 8
    high_policy = ConcurrentExecutionWorkPolicy(max_concurrency=65)
    assert high_policy.max_concurrency == 65


def test_graph_runner_retry_eligibility_propagates_global_deadline() -> None:
    source = _GRAPH_RUNNER.read_text(encoding="utf-8")
    assert "global_deadline_monotonic=peek_active_execution_global_deadline_monotonic()" in source


def test_execution_capacity_policy_contract_exists() -> None:
    from intergrax.contracts.execution_capacity_admission import (
        ExecutionCapacityPolicy,
    )

    assert ExecutionCapacityPolicy(max_concurrent_root_executions=4).max_concurrent_root_executions == 4


def test_execution_runtime_acquires_capacity_before_delegate() -> None:
    runtime_source = (
        _REPO_ROOT / "intergrax" / "runtime" / "execution" / "runtime.py"
    ).read_text(encoding="utf-8")
    assert "_execution_capacity_admission.acquire" in runtime_source
    assert "await capacity_permit.release()" in runtime_source
    assert runtime_source.index("_execution_capacity_admission.acquire") < runtime_source.index(
        "boundary.execute"
    )


def test_orchestration_scheduling_policy_carries_submission_concurrency() -> None:
    policy = OrchestrationSchedulingPolicy(max_concurrency=8)
    assert policy.max_concurrency == 8
