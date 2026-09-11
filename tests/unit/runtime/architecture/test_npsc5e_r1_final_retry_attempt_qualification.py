# © Artur Czarnecki. All rights reserved.

"""NPSC-5E/R1 Final — cross-layer canonical execution-attempt retry qualification & freeze."""

from __future__ import annotations

import ast
import importlib
import re
import subprocess
import threading
from pathlib import Path

import pytest

from intergrax.contracts.attempt_lifecycle import AttemptTransitionReason
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    peek_active_execution_identity,
    reset_active_execution_identity,
)
from intergrax.contracts.execution_lineage import (
    ExecutionLineageAttemptClosureKind,
    build_execution_lineage_attempt_scope,
)
from intergrax.contracts.execution_retry import (
    BackoffKind,
    BackoffPolicyConfig,
    ExecutionFailureKind,
    ExecutionRetryAction,
    ExecutionRetryEligibilityRequest,
)
from intergrax.contracts.execution_terminal import ExecutionTerminalOutcome
from intergrax.contracts.lease_claim import StaleClaimError
from intergrax.contracts.resilience_policy import FailureResponse
from intergrax.runtime.cancellation.coordinator import CancellationCoordinator
from intergrax.runtime.execution.attempt_lifecycle import (
    AttemptLifecycleService,
    InMemoryAttemptLifecycleStore,
)
from intergrax.runtime.execution.lineage.persistence import InMemoryExecutionLineagePersistence
from intergrax.runtime.execution.retry import (
    ExecutionAttemptRetryService,
    classify_execution_failure,
    classify_from_failure_response,
    compute_backoff_delay,
    evaluate_execution_retry_eligibility,
)
from intergrax.runtime.reliability.step_retry_budget import StepRetryBudget
from tests.unit.runtime.execution.lineage.lineage_test_helpers import register_v1_attempt

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]

R1_IMPLEMENTATION_SHA = "0dda823a2682551d0c8047fc6be288731a640ed2"
NPSC_5D_FROZEN_SHA = "a4a1faca01cd5004e372f235132184a84aa5a6bd"
NPSC_5E_LINEAGE_BASELINE_SHA = "a72c9b568c61e28180756059ae48a99fb56eaa19"
P0A_QUALIFICATION_SHA = "7a2fcc6415f8d56234977e98f8c7f3329b5e8e96"

_RETRY_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "retry"
_GRAPH_RUNNER_PATH = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "orchestration" / "graph_runner.py"
_ATTEMPT_LIFECYCLE_PATH = (
    _REPO_ROOT / "intergrax" / "runtime" / "execution" / "attempt_lifecycle" / "service.py"
)

_MINT_RETRY_ALLOWED_FILES = frozenset(
    {
        "intergrax/runtime/execution/identity_authority.py",
        "intergrax/runtime/execution/attempt_lifecycle/service.py",
    },
)

_FORBIDDEN_RETRY_RUNTIME_NAMES = (
    "RetryRuntime",
    "ExecutionRecoveryRuntime",
    "AlternateAttemptExecutor",
    "EnterpriseRetryRecoveryManager",
    "UnifiedRecoveryEverythingService",
)

_RESELECTION_NAMES = (
    "AgentDiscoveryStrategy",
    "CapabilityMatcher",
    "AgentSelectionStrategy",
)

_REFLECTION_PATTERN = re.compile(r"\b(getattr|setattr|hasattr)\(")

_FROZEN_REGRESSION_MODULES = (
    "tests.unit.runtime.architecture.test_npsc5e_r1_execution_retry_attempt_semantics",
    "tests.unit.runtime.architecture.test_npsc5e_p0a_execution_lineage_baseline_qualification",
    "tests.unit.runtime.architecture.test_npsc5d_r1_final_qualification",
    "tests.unit.runtime.architecture.test_npsc5d_r2_final_qualification",
    "tests.unit.runtime.architecture.test_npsc5d_r3_final_qualification",
    "tests.unit.runtime.architecture.test_npsc5d_final_multi_agent_governance_qualification",
    "tests.unit.runtime.architecture.test_npsc5a_multi_agent_coordination_gate",
    "tests.unit.runtime.architecture.test_npsc5a_coordination_delegation_e2e",
    "tests.unit.runtime.architecture.test_npsc5b_final_production_fanout_fanin_qualification",
    "tests.unit.runtime.architecture.test_npsc5c_coordination_intent_gate",
    "tests.unit.runtime.architecture.test_npsc5c_decision_execution_e2e",
)

_NON_RETRYABLE_KINDS = frozenset(
    {
        ExecutionFailureKind.GOVERNANCE_DENIED,
        ExecutionFailureKind.AUTHORITY_DENIED,
        ExecutionFailureKind.TRUST_DENIED,
        ExecutionFailureKind.CONTRACT_ERROR,
        ExecutionFailureKind.UNKNOWN,
        ExecutionFailureKind.UNKNOWN_UNSAFE,
        ExecutionFailureKind.NON_RETRYABLE_PERMANENT,
        ExecutionFailureKind.TERMINAL_DENY,
        ExecutionFailureKind.TERMINAL_SUCCESS,
        ExecutionFailureKind.BUDGET_EXHAUSTED,
        ExecutionFailureKind.DEADLINE_EXCEEDED,
        ExecutionFailureKind.CANCELLED,
    },
)

_RETRYABLE_KINDS = frozenset(
    {
        ExecutionFailureKind.RETRYABLE_TRANSIENT,
        ExecutionFailureKind.RETRYABLE_TIMEOUT,
    },
)


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _eligibility_request(
    kind: ExecutionFailureKind,
    *,
    attempt_number: int = 1,
    max_attempts: int = 3,
    cancelled: bool = False,
    terminal_outcome: ExecutionTerminalOutcome | None = None,
    deadline: float | None = None,
    now: float | None = None,
    backoff: float = 0.0,
    unknown_side_effect: bool = False,
    idempotency: bool = False,
) -> ExecutionRetryEligibilityRequest:
    return ExecutionRetryEligibilityRequest(
        classification=classify_execution_failure(
            kind=kind,
            has_unknown_side_effect=unknown_side_effect,
        ),
        attempt_number=attempt_number,
        max_attempts=max_attempts,
        cancelled=cancelled,
        terminal_outcome=terminal_outcome,
        global_deadline_monotonic=deadline,
        now_monotonic=now,
        proposed_backoff_seconds=backoff,
        side_effect_idempotency_guaranteed=idempotency,
    )


def _retry_service(
    *,
    lineage: bool = True,
) -> tuple[ExecutionAttemptRetryService, AttemptLifecycleService, InMemoryExecutionLineagePersistence | None]:
    lifecycle = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    persistence = InMemoryExecutionLineagePersistence() if lineage else None
    return (
        ExecutionAttemptRetryService(lifecycle, lineage_persistence=persistence),
        lifecycle,
        persistence,
    )


def _collect_mint_retry_outside_authority() -> list[str]:
    violations: list[str] = []
    scan_roots = (
        _REPO_ROOT / "intergrax" / "runtime" / "execution",
        _REPO_ROOT / "intergrax" / "runtime" / "nexus",
    )
    for root in scan_roots:
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(_REPO_ROOT).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if _call_name(node.func) != "mint_retry_attempt_id":
                    continue
                if rel in _MINT_RETRY_ALLOWED_FILES:
                    continue
                violations.append(f"{rel}:{node.lineno}")
    return violations


def _graph_runner_retry_method_source() -> str:
    source = _GRAPH_RUNNER_PATH.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(_GRAPH_RUNNER_PATH))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_transition_attempt_for_retry":
            lines = source.splitlines()
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise AssertionError("_transition_attempt_for_retry not found")


@pytest.mark.gate
def test_r1_final_provenance_commits_exist() -> None:
    for sha in (
        R1_IMPLEMENTATION_SHA,
        NPSC_5D_FROZEN_SHA,
        NPSC_5E_LINEAGE_BASELINE_SHA,
        P0A_QUALIFICATION_SHA,
    ):
        result = subprocess.run(
            ["git", "rev-parse", "--verify", f"{sha}^{{commit}}"],
            cwd=_REPO_ROOT,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, sha


@pytest.mark.gate
def test_r1_final_static_canonical_ownership_and_wiring() -> None:
    lifecycle_source = _ATTEMPT_LIFECYCLE_PATH.read_text(encoding="utf-8-sig")
    graph_source = _GRAPH_RUNNER_PATH.read_text(encoding="utf-8-sig")
    assert "transition_to_next_attempt" in lifecycle_source
    assert "mint_retry_attempt_id()" in lifecycle_source
    assert "ExecutionAttemptRetryService" in graph_source
    assert "_transition_attempt_for_retry" in graph_source
    assert _collect_mint_retry_outside_authority() == []


@pytest.mark.gate
def test_r1_final_no_second_retry_runtime_or_scheduler() -> None:
    violations: list[str] = []
    for path in (_REPO_ROOT / "intergrax").rglob("*.py"):
        source = path.read_text(encoding="utf-8-sig")
        for name in _FORBIDDEN_RETRY_RUNTIME_NAMES:
            if name in source:
                violations.append(f"{path.relative_to(_REPO_ROOT)}:{name}")
    assert violations == []


@pytest.mark.gate
def test_r1_final_retry_path_no_reselection() -> None:
    retry_source = _graph_runner_retry_method_source()
    for name in _RESELECTION_NAMES:
        assert name not in retry_source


@pytest.mark.gate
def test_r1_final_transport_step_agent_retry_distinct() -> None:
    transport = (_REPO_ROOT / "intergrax" / "queueing" / "worker" / "retry_policy.py").read_text(
        encoding="utf-8-sig",
    )
    retry_engine = (_REPO_ROOT / "intergrax" / "runtime" / "nexus" / "retry" / "retry_engine.py").read_text(
        encoding="utf-8-sig",
    )
    step_budget = (_REPO_ROOT / "intergrax" / "runtime" / "reliability" / "step_retry_budget.py").read_text(
        encoding="utf-8-sig",
    )
    assert "AttemptId" not in transport
    assert "mint_attempt_id" not in retry_engine
    assert "transition_to_next_attempt" not in retry_engine
    assert "AttemptId" not in step_budget
    budget = StepRetryBudget(max_retries=2)
    assert budget.can_retry()


@pytest.mark.gate
def test_r1_final_hitl_distinct_from_execution_retry() -> None:
    kind = classify_from_failure_response(FailureResponse.REQUEST_HUMAN)
    assert kind is ExecutionFailureKind.GOVERNANCE_DENIED
    result = evaluate_execution_retry_eligibility(_eligibility_request(kind))
    assert result.action is ExecutionRetryAction.FAIL


@pytest.mark.gate
@pytest.mark.parametrize("kind", list(ExecutionFailureKind))
def test_r1_final_all_execution_failure_kinds_classified(kind: ExecutionFailureKind) -> None:
    result = evaluate_execution_retry_eligibility(_eligibility_request(kind))
    if kind in _RETRYABLE_KINDS:
        assert result.action is ExecutionRetryAction.RETRY
    elif kind is ExecutionFailureKind.CANCELLED:
        assert result.action is ExecutionRetryAction.CANCEL
    else:
        assert result.action is ExecutionRetryAction.FAIL


@pytest.mark.gate
@pytest.mark.parametrize("kind", list(BackoffKind))
def test_r1_final_all_backoff_kinds_bounded(kind: BackoffKind) -> None:
    config = BackoffPolicyConfig(
        kind=kind,
        base_delay_seconds=2.0,
        multiplier=2.0,
        max_delay_seconds=5.0,
        jitter_ratio=0.5,
    )
    delay = compute_backoff_delay(attempt_number=5, config=config)
    assert 0.0 <= delay <= config.max_delay_seconds
    retry_after = compute_backoff_delay(
        attempt_number=5,
        config=config,
        retry_after_seconds=100.0,
    )
    assert retry_after <= config.max_delay_seconds


@pytest.mark.gate
@pytest.mark.parametrize(
    ("terminal_outcome", "expected"),
    [
        (ExecutionTerminalOutcome.COMPLETED, ExecutionRetryAction.FAIL),
        (ExecutionTerminalOutcome.CANCELLED, ExecutionRetryAction.CANCEL),
        (ExecutionTerminalOutcome.FAILED, ExecutionRetryAction.FAIL),
    ],
)
def test_r1_final_terminal_matrix(
    terminal_outcome: ExecutionTerminalOutcome,
    expected: ExecutionRetryAction,
) -> None:
    result = evaluate_execution_retry_eligibility(
        _eligibility_request(
            ExecutionFailureKind.RETRYABLE_TRANSIENT,
            terminal_outcome=terminal_outcome,
        ),
    )
    assert result.action is expected


@pytest.mark.gate
def test_r1_final_cross_layer_transient_fail_then_success() -> None:
    service, lifecycle, persistence = _retry_service(lineage=True)
    assert persistence is not None
    tenant_id = "tenant-a"
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    scope = build_execution_lineage_attempt_scope(
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a1,
    )
    register_v1_attempt(persistence, scope)
    lifecycle.record_initial_attempt(tenant_id=tenant_id, run_id=run_id, attempt_id=attempt_a1)
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_a1, execution_id=mint_execution_id())
    try:
        transition = service.transition_for_retry(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            expected_attempt_id=attempt_a1,
            request=_eligibility_request(
                ExecutionFailureKind.RETRYABLE_TRANSIENT,
                attempt_number=1,
                max_attempts=3,
            ),
        )
        assert transition is not None
        assert transition.run_id == run_id
        assert transition.previous_attempt_id == attempt_a1
        assert transition.active_attempt_id != attempt_a1
        assert transition.generation == 2
        assert peek_active_execution_identity() == (run_id, transition.active_attempt_id)
        seal = persistence.read_seal(scope)
        assert seal is not None
        assert seal.closure_kind is ExecutionLineageAttemptClosureKind.RETRY_SUPERSEDED
        blocked = service.transition_for_retry(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            expected_attempt_id=transition.active_attempt_id,
            request=_eligibility_request(ExecutionFailureKind.NON_RETRYABLE_PERMANENT),
        )
        assert blocked is None
    finally:
        reset_active_execution_identity(token)


@pytest.mark.gate
def test_r1_final_permanent_failure_no_successor() -> None:
    service, lifecycle, _ = _retry_service(lineage=False)
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    lifecycle.record_initial_attempt(tenant_id="tenant-a", run_id=run_id, attempt_id=attempt_a1)
    assert (
        service.transition_for_retry(
            tenant_id="tenant-a",
            task_id=mint_task_id(),
            run_id=run_id,
            expected_attempt_id=attempt_a1,
            request=_eligibility_request(ExecutionFailureKind.NON_RETRYABLE_PERMANENT),
        )
        is None
    )


@pytest.mark.gate
def test_r1_final_max_attempts_exhaustion_off_by_one() -> None:
    service, lifecycle, _ = _retry_service(lineage=False)
    tenant_id = "tenant-a"
    run_id = mint_run_id()
    attempt = mint_attempt_id()
    lifecycle.record_initial_attempt(tenant_id=tenant_id, run_id=run_id, attempt_id=attempt)
    attempts = [attempt]
    for attempt_number in (1, 2):
        transition = service.transition_for_retry(
            tenant_id=tenant_id,
            task_id=mint_task_id(),
            run_id=run_id,
            expected_attempt_id=attempts[-1],
            request=_eligibility_request(
                ExecutionFailureKind.RETRYABLE_TRANSIENT,
                attempt_number=attempt_number,
                max_attempts=3,
            ),
        )
        assert transition is not None
        attempts.append(transition.active_attempt_id)
    blocked = service.transition_for_retry(
        tenant_id=tenant_id,
        task_id=mint_task_id(),
        run_id=run_id,
        expected_attempt_id=attempts[-1],
        request=_eligibility_request(
            ExecutionFailureKind.RETRYABLE_TRANSIENT,
            attempt_number=3,
            max_attempts=3,
        ),
    )
    assert blocked is None
    assert len(attempts) == 3


@pytest.mark.gate
def test_r1_final_cancelled_blocks_retry() -> None:
    result = evaluate_execution_retry_eligibility(
        _eligibility_request(ExecutionFailureKind.RETRYABLE_TRANSIENT, cancelled=True),
    )
    assert result.action is ExecutionRetryAction.CANCEL
    service, lifecycle, _ = _retry_service(lineage=False)
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    lifecycle.record_initial_attempt(tenant_id="tenant-a", run_id=run_id, attempt_id=attempt_a1)
    assert (
        service.transition_for_retry(
            tenant_id="tenant-a",
            task_id=mint_task_id(),
            run_id=run_id,
            expected_attempt_id=attempt_a1,
            request=_eligibility_request(ExecutionFailureKind.RETRYABLE_TRANSIENT, cancelled=True),
        )
        is None
    )


@pytest.mark.gate
def test_r1_final_deadline_blocks_retry() -> None:
    result = evaluate_execution_retry_eligibility(
        _eligibility_request(
            ExecutionFailureKind.RETRYABLE_TRANSIENT,
            deadline=100.5,
            now=100.0,
            backoff=1.0,
        ),
    )
    assert result.action is ExecutionRetryAction.FAIL
    assert result.reason == "global_deadline_exceeded"


@pytest.mark.gate
@pytest.mark.parametrize(
    "kind",
    [
        ExecutionFailureKind.GOVERNANCE_DENIED,
        ExecutionFailureKind.AUTHORITY_DENIED,
        ExecutionFailureKind.TRUST_DENIED,
    ],
)
def test_r1_final_deny_kinds_block_transition(kind: ExecutionFailureKind) -> None:
    service, lifecycle, _ = _retry_service(lineage=False)
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    lifecycle.record_initial_attempt(tenant_id="tenant-a", run_id=run_id, attempt_id=attempt_a1)
    assert (
        service.transition_for_retry(
            tenant_id="tenant-a",
            task_id=mint_task_id(),
            run_id=run_id,
            expected_attempt_id=attempt_a1,
            request=_eligibility_request(kind),
        )
        is None
    )


@pytest.mark.gate
def test_r1_final_unknown_fail_closed() -> None:
    result = evaluate_execution_retry_eligibility(_eligibility_request(ExecutionFailureKind.UNKNOWN))
    assert result.action is ExecutionRetryAction.FAIL
    assert result.reason == "unknown_fail_closed"


@pytest.mark.gate
def test_r1_final_unknown_unsafe_no_blind_retry() -> None:
    result = evaluate_execution_retry_eligibility(
        _eligibility_request(
            ExecutionFailureKind.RETRYABLE_TRANSIENT,
            unknown_side_effect=True,
            idempotency=False,
        ),
    )
    assert result.action is ExecutionRetryAction.FAIL


@pytest.mark.gate
def test_r1_final_duplicate_retry_idempotent() -> None:
    service, lifecycle, _ = _retry_service(lineage=False)
    tenant_id = "tenant-a"
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    lifecycle.record_initial_attempt(tenant_id=tenant_id, run_id=run_id, attempt_id=attempt_a1)
    request = _eligibility_request(ExecutionFailureKind.RETRYABLE_TRANSIENT)
    first = service.transition_for_retry(
        tenant_id=tenant_id,
        task_id=mint_task_id(),
        run_id=run_id,
        expected_attempt_id=attempt_a1,
        request=request,
    )
    second = service.transition_for_retry(
        tenant_id=tenant_id,
        task_id=mint_task_id(),
        run_id=run_id,
        expected_attempt_id=attempt_a1,
        request=request,
    )
    assert first is not None
    assert second is None


@pytest.mark.gate
def test_r1_final_concurrent_retry_exactly_one_successor() -> None:
    lifecycle = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    service = ExecutionAttemptRetryService(lifecycle)
    tenant_id = "tenant-a"
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    lifecycle.record_initial_attempt(tenant_id=tenant_id, run_id=run_id, attempt_id=attempt_a1)
    barrier = threading.Barrier(2)
    results: list[str] = []
    request = _eligibility_request(ExecutionFailureKind.RETRYABLE_TRANSIENT)

    def worker() -> None:
        barrier.wait()
        transition = service.transition_for_retry(
            tenant_id=tenant_id,
            task_id=mint_task_id(),
            run_id=run_id,
            expected_attempt_id=attempt_a1,
            request=request,
        )
        if transition is not None:
            results.append(str(transition.active_attempt_id))

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert len(results) == 1


@pytest.mark.gate
def test_r1_final_attempt_fork_blocked() -> None:
    lifecycle = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    service = ExecutionAttemptRetryService(lifecycle)
    tenant_id = "tenant-a"
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    lifecycle.record_initial_attempt(tenant_id=tenant_id, run_id=run_id, attempt_id=attempt_a1)
    first = service.transition_for_retry(
        tenant_id=tenant_id,
        task_id=mint_task_id(),
        run_id=run_id,
        expected_attempt_id=attempt_a1,
        request=_eligibility_request(ExecutionFailureKind.RETRYABLE_TRANSIENT),
    )
    assert first is not None
    fork = service.transition_for_retry(
        tenant_id=tenant_id,
        task_id=mint_task_id(),
        run_id=run_id,
        expected_attempt_id=attempt_a1,
        request=_eligibility_request(ExecutionFailureKind.RETRYABLE_TRANSIENT),
    )
    assert fork is None


@pytest.mark.gate
def test_r1_final_attempt_skip_blocked() -> None:
    lifecycle = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    service = ExecutionAttemptRetryService(lifecycle)
    tenant_id = "tenant-a"
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    lifecycle.record_initial_attempt(tenant_id=tenant_id, run_id=run_id, attempt_id=attempt_a1)
    first = service.transition_for_retry(
        tenant_id=tenant_id,
        task_id=mint_task_id(),
        run_id=run_id,
        expected_attempt_id=attempt_a1,
        request=_eligibility_request(ExecutionFailureKind.RETRYABLE_TRANSIENT, attempt_number=1, max_attempts=3),
    )
    assert first is not None
    with pytest.raises(StaleClaimError):
        lifecycle.transition_to_next_attempt(
            tenant_id=tenant_id,
            run_id=run_id,
            expected_attempt_id=attempt_a1,
            reason=AttemptTransitionReason.RETRY,
        )


@pytest.mark.gate
def test_r1_final_cancellation_coordinator_precedence() -> None:
    from intergrax.runtime.cancellation.coordinator import CANCELLATION_REQUESTED_KEY

    assert CancellationCoordinator.is_requested({CANCELLATION_REQUESTED_KEY: True})


@pytest.mark.gate
def test_r1_final_retry_module_no_reflection() -> None:
    violations: list[str] = []
    for path in _RETRY_ROOT.rglob("*.py"):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
            if _REFLECTION_PATTERN.search(line):
                violations.append(f"{rel}:{lineno}")
    assert violations == []


@pytest.mark.gate
@pytest.mark.parametrize("module_name", _FROZEN_REGRESSION_MODULES)
def test_r1_final_frozen_regression_module_importable(module_name: str) -> None:
    importlib.import_module(module_name)
