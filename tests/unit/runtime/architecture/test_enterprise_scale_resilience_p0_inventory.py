# © Artur Czarnecki. All rights reserved.

"""P0 — enterprise execution scale & resilience static inventory gates."""

from __future__ import annotations

import ast
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
_CONCURRENT_WORK = (
    _REPO_ROOT / "intergrax" / "runtime" / "execution" / "concurrent_execution_work.py"
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


def test_concurrent_execution_work_has_no_platform_width_constant() -> None:
    source = _CONCURRENT_WORK.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assigned_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "MAX_CONCURRENT_EXECUTION_WORK" not in assigned_names


def test_graph_runner_retry_eligibility_request_documents_deadline_gap() -> None:
    """Documented gap: Nexus graph retry path does not pass global_deadline_monotonic."""
    source = _GRAPH_RUNNER.read_text(encoding="utf-8")
    assert "ExecutionRetryEligibilityRequest(" in source
    assert "global_deadline_monotonic" not in source


def test_orchestration_scheduling_policy_carries_submission_concurrency() -> None:
    policy = OrchestrationSchedulingPolicy(max_concurrency=8)
    assert policy.max_concurrency == 8
