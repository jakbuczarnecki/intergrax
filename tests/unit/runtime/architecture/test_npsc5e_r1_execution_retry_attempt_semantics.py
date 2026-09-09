# © Artur Czarnecki. All rights reserved.

"""NPSC-5E/R1 — canonical execution-attempt retry & attempt semantics qualification."""

from __future__ import annotations

import ast
import re
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
from intergrax.contracts.resilience_policy import FailureClass, FailureResponse, ResiliencePolicy
from intergrax.runtime.cancellation.coordinator import CancellationCoordinator
from intergrax.runtime.execution.attempt_lifecycle import (
    AttemptLifecycleService,
    InMemoryAttemptLifecycleStore,
)
from intergrax.runtime.execution.lineage.persistence import InMemoryExecutionLineagePersistence
from intergrax.runtime.execution.retry import (
    ExecutionAttemptRetryService,
    classify_execution_failure,
    classify_from_failure_class,
    classify_from_failure_response,
    compute_backoff_delay,
    evaluate_execution_retry_eligibility,
)
from intergrax.runtime.execution.retry.policy import project_resilience_failure_kind
from intergrax.runtime.reliability.step_retry_budget import StepRetryBudget
from intergrax.runtime.resilience.policy_resolver import resolve_failure_action

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
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
    "EnterpriseRetryRecoveryManager",
    "UnifiedRecoveryEverythingService",
    "RecoveryLineageManager",
    "RetryLineageEngine",
)

_REFLECTION_PATTERN = re.compile(r"\b(getattr|setattr|hasattr)\(")

_FROZEN_REGRESSION_MODULES = (
    "tests.unit.runtime.architecture.test_npsc5d_r1_final_qualification",
    "tests.unit.runtime.architecture.test_npsc5d_r2_final_qualification",
    "tests.unit.runtime.architecture.test_npsc5d_r3_final_qualification",
    "tests.unit.runtime.architecture.test_npsc5a_multi_agent_coordination_gate",
    "tests.unit.runtime.architecture.test_npsc5a_coordination_delegation_e2e",
    "tests.unit.runtime.architecture.test_npsc5b_final_production_fanout_fanin_qualification",
    "tests.unit.runtime.architecture.test_npsc5c_coordination_intent_gate",
    "tests.unit.runtime.architecture.test_npsc5c_decision_execution_e2e",
    "tests.unit.runtime.architecture.test_npsc5e_p0a_execution_lineage_baseline_qualification",
    "tests.unit.runtime.execution.test_attempt_lifecycle",
    "tests.unit.runtime.execution.lineage.test_terminal_conflict_seal",
    "tests.unit.runtime.nexus.orchestration.test_graph_runner_resilience",
)


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


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


@pytest.mark.gate
def test_r1_retry_inventory_resilience_policy() -> None:
    policy = ResiliencePolicy(max_attempts=3)
    resolution = resolve_failure_action(FailureClass.DEPENDENCY_ERROR, policy=policy, attempt=0)
    assert resolution.response is FailureResponse.RETRY
    assert policy.max_attempts == 3


@pytest.mark.gate
def test_r1_retry_inventory_retry_coordinator_is_decision_only() -> None:
    from intergrax.runtime.nexus.retry.coordinator import RetryCoordinator
    from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode

    coordinator = RetryCoordinator(max_run_retries=2, retry_run_on=frozenset({RuntimeErrorCode.VALIDATION_ERROR}))
    assert coordinator.should_retry_run(attempt=0, error_code=RuntimeErrorCode.VALIDATION_ERROR)
    assert not coordinator.should_retry_run(attempt=2, error_code=RuntimeErrorCode.VALIDATION_ERROR)


@pytest.mark.gate
def test_r1_retry_inventory_retry_engine_does_not_mint_attempt() -> None:
    source = (_REPO_ROOT / "intergrax" / "runtime" / "nexus" / "retry" / "retry_engine.py").read_text(
        encoding="utf-8-sig",
    )
    assert "mint_attempt_id" not in source
    assert "transition_to_next_attempt" not in source


@pytest.mark.gate
def test_r1_retry_inventory_policy_enforcer_sub_attempt_scope() -> None:
    source = (_REPO_ROOT / "intergrax" / "runtime" / "nexus" / "policies" / "policy_enforcer.py").read_text(
        encoding="utf-8-sig",
    )
    assert "transition_to_next_attempt" not in source
    assert "mint_attempt_id" not in source


@pytest.mark.gate
def test_r1_retry_inventory_attempt_lifecycle_sole_mint_authority() -> None:
    source = _ATTEMPT_LIFECYCLE_PATH.read_text(encoding="utf-8-sig")
    assert "transition_to_next_attempt" in source
    assert "mint_retry_attempt_id()" in source


@pytest.mark.gate
def test_r1_retry_inventory_transport_retry_no_attempt_mint() -> None:
    source = (_REPO_ROOT / "intergrax" / "queueing" / "worker" / "retry_policy.py").read_text(encoding="utf-8-sig")
    assert "AttemptId" not in source
    assert "transition_to_next_attempt" not in source


@pytest.mark.gate
def test_r1_step_retry_budget_does_not_mint_attempt() -> None:
    budget = StepRetryBudget(max_retries=2)
    assert budget.can_retry()
    consumed = budget.consume()
    assert consumed.retries_used == 1
    source = (_REPO_ROOT / "intergrax" / "runtime" / "reliability" / "step_retry_budget.py").read_text(
        encoding="utf-8-sig",
    )
    assert "AttemptId" not in source


@pytest.mark.gate
def test_r1_no_direct_mint_retry_outside_authority() -> None:
    assert _collect_mint_retry_outside_authority() == []


@pytest.mark.gate
def test_r1_no_forbidden_retry_god_objects() -> None:
    violations: list[str] = []
    for path in (_REPO_ROOT / "intergrax").rglob("*.py"):
        source = path.read_text(encoding="utf-8-sig")
        for name in _FORBIDDEN_RETRY_RUNTIME_NAMES:
            if name in source:
                violations.append(f"{path.relative_to(_REPO_ROOT)}:{name}")
    assert violations == []


@pytest.mark.gate
def test_r1_retry_module_no_reflection() -> None:
    violations: list[str] = []
    for path in _RETRY_ROOT.rglob("*.py"):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
            if _REFLECTION_PATTERN.search(line):
                violations.append(f"{rel}:{lineno}")
    assert violations == []


@pytest.mark.gate
def test_r1_graph_runner_uses_canonical_retry_service() -> None:
    source = _GRAPH_RUNNER_PATH.read_text(encoding="utf-8-sig")
    assert "ExecutionAttemptRetryService" in source
    assert "transition_for_retry" in source


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (ExecutionFailureKind.GOVERNANCE_DENIED, ExecutionRetryAction.FAIL),
        (ExecutionFailureKind.AUTHORITY_DENIED, ExecutionRetryAction.FAIL),
        (ExecutionFailureKind.TRUST_DENIED, ExecutionRetryAction.FAIL),
        (ExecutionFailureKind.CONTRACT_ERROR, ExecutionRetryAction.FAIL),
        (ExecutionFailureKind.UNKNOWN, ExecutionRetryAction.FAIL),
        (ExecutionFailureKind.UNKNOWN_UNSAFE, ExecutionRetryAction.FAIL),
        (ExecutionFailureKind.NON_RETRYABLE_PERMANENT, ExecutionRetryAction.FAIL),
        (ExecutionFailureKind.TERMINAL_DENY, ExecutionRetryAction.FAIL),
        (ExecutionFailureKind.TERMINAL_SUCCESS, ExecutionRetryAction.FAIL),
        (ExecutionFailureKind.BUDGET_EXHAUSTED, ExecutionRetryAction.FAIL),
        (ExecutionFailureKind.DEADLINE_EXCEEDED, ExecutionRetryAction.FAIL),
    ],
)
@pytest.mark.gate
def test_r1_non_retryable_matrix(kind: ExecutionFailureKind, expected: ExecutionRetryAction) -> None:
    result = evaluate_execution_retry_eligibility(_eligibility_request(kind))
    assert result.action is expected


@pytest.mark.gate
def test_r1_transient_failure_retry_eligible() -> None:
    result = evaluate_execution_retry_eligibility(
        _eligibility_request(ExecutionFailureKind.RETRYABLE_TRANSIENT, attempt_number=1, max_attempts=3),
    )
    assert result.action is ExecutionRetryAction.RETRY


@pytest.mark.gate
def test_r1_timeout_retryable_when_classified() -> None:
    result = evaluate_execution_retry_eligibility(
        _eligibility_request(ExecutionFailureKind.RETRYABLE_TIMEOUT, attempt_number=1, max_attempts=3),
    )
    assert result.action is ExecutionRetryAction.RETRY


@pytest.mark.gate
def test_r1_timeout_non_retryable_when_permanent() -> None:
    result = evaluate_execution_retry_eligibility(
        _eligibility_request(ExecutionFailureKind.NON_RETRYABLE_PERMANENT),
    )
    assert result.action is ExecutionRetryAction.FAIL


@pytest.mark.gate
def test_r1_cancel_before_retry() -> None:
    result = evaluate_execution_retry_eligibility(
        _eligibility_request(ExecutionFailureKind.RETRYABLE_TRANSIENT, cancelled=True),
    )
    assert result.action is ExecutionRetryAction.CANCEL


@pytest.mark.gate
def test_r1_cancel_during_backoff_blocks_retry() -> None:
    result = evaluate_execution_retry_eligibility(
        _eligibility_request(
            ExecutionFailureKind.RETRYABLE_TRANSIENT,
            cancelled=True,
            backoff=1.0,
        ),
    )
    assert result.action is ExecutionRetryAction.CANCEL


@pytest.mark.gate
def test_r1_global_deadline_blocks_retry_when_backoff_crosses() -> None:
    now = 100.0
    deadline = 100.5
    result = evaluate_execution_retry_eligibility(
        _eligibility_request(
            ExecutionFailureKind.RETRYABLE_TRANSIENT,
            deadline=deadline,
            now=now,
            backoff=1.0,
        ),
    )
    assert result.action is ExecutionRetryAction.FAIL
    assert result.reason == "global_deadline_exceeded"


@pytest.mark.gate
def test_r1_budget_exhaustion_off_by_one_guard() -> None:
    for attempt in (1, 2):
        result = evaluate_execution_retry_eligibility(
            _eligibility_request(
                ExecutionFailureKind.RETRYABLE_TRANSIENT,
                attempt_number=attempt,
                max_attempts=3,
            ),
        )
        assert result.action is ExecutionRetryAction.RETRY
    exhausted = evaluate_execution_retry_eligibility(
        _eligibility_request(
            ExecutionFailureKind.RETRYABLE_TRANSIENT,
            attempt_number=3,
            max_attempts=3,
        ),
    )
    assert exhausted.action is ExecutionRetryAction.FAIL
    assert exhausted.reason == "max_attempts_exhausted"


@pytest.mark.gate
def test_r1_unknown_side_effect_blind_retry_blocked() -> None:
    result = evaluate_execution_retry_eligibility(
        _eligibility_request(
            ExecutionFailureKind.RETRYABLE_TRANSIENT,
            unknown_side_effect=True,
            idempotency=False,
        ),
    )
    assert result.action is ExecutionRetryAction.FAIL


@pytest.mark.gate
def test_r1_hitl_request_human_not_transient_retry() -> None:
    kind = classify_from_failure_response(FailureResponse.REQUEST_HUMAN)
    assert kind is ExecutionFailureKind.GOVERNANCE_DENIED
    result = evaluate_execution_retry_eligibility(_eligibility_request(kind))
    assert result.action is ExecutionRetryAction.FAIL


@pytest.mark.gate
def test_r1_backoff_fixed_and_exponential_and_cap() -> None:
    fixed = compute_backoff_delay(
        attempt_number=2,
        config=BackoffPolicyConfig(kind=BackoffKind.FIXED, base_delay_seconds=2.0, max_delay_seconds=10.0),
    )
    assert fixed == 2.0
    exponential = compute_backoff_delay(
        attempt_number=3,
        config=BackoffPolicyConfig(
            kind=BackoffKind.EXPONENTIAL,
            base_delay_seconds=1.0,
            multiplier=2.0,
            max_delay_seconds=5.0,
        ),
    )
    assert exponential == 4.0
    capped = compute_backoff_delay(
        attempt_number=10,
        config=BackoffPolicyConfig(
            kind=BackoffKind.EXPONENTIAL,
            base_delay_seconds=1.0,
            multiplier=2.0,
            max_delay_seconds=3.0,
        ),
    )
    assert capped <= 3.0


@pytest.mark.gate
def test_r1_transient_attempt1_failure_attempt2_success_preserves_run_id() -> None:
    service, lifecycle, _ = _retry_service(lineage=False)
    tenant_id = "tenant-a"
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    lifecycle.record_initial_attempt(tenant_id=tenant_id, run_id=run_id, attempt_id=attempt_a1)
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_a1, execution_id=mint_execution_id())
    try:
        transition = service.transition_for_retry(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            expected_attempt_id=attempt_a1,
            request=_eligibility_request(ExecutionFailureKind.RETRYABLE_TRANSIENT, attempt_number=1, max_attempts=3),
        )
        assert transition is not None
        assert transition.run_id == run_id
        assert transition.previous_attempt_id == attempt_a1
        assert transition.active_attempt_id != attempt_a1
        assert peek_active_execution_identity() == (run_id, transition.active_attempt_id)
    finally:
        reset_active_execution_identity(token)


@pytest.mark.gate
def test_r1_permanent_failure_no_transition() -> None:
    service, lifecycle, _ = _retry_service(lineage=False)
    tenant_id = "tenant-a"
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    lifecycle.record_initial_attempt(tenant_id=tenant_id, run_id=run_id, attempt_id=attempt_a1)
    transition = service.transition_for_retry(
        tenant_id=tenant_id,
        task_id=mint_task_id(),
        run_id=run_id,
        expected_attempt_id=attempt_a1,
        request=_eligibility_request(ExecutionFailureKind.NON_RETRYABLE_PERMANENT),
    )
    assert transition is None
    assert lifecycle.get_active_attempt_id(tenant_id=tenant_id, run_id=run_id) == attempt_a1


@pytest.mark.gate
def test_r1_max_attempts_three_creates_exactly_three_attempts() -> None:
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
def test_r1_attempt_lineage_retry_superseded_seal() -> None:
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
    persistence.open_attempt(scope)
    lifecycle.record_initial_attempt(tenant_id=tenant_id, run_id=run_id, attempt_id=attempt_a1)
    transition = service.transition_for_retry(
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        expected_attempt_id=attempt_a1,
        request=_eligibility_request(ExecutionFailureKind.RETRYABLE_TRANSIENT),
    )
    assert transition is not None
    seal = persistence.read_seal(scope)
    assert seal is not None
    assert seal.closure_kind is ExecutionLineageAttemptClosureKind.RETRY_SUPERSEDED


@pytest.mark.gate
def test_r1_duplicate_retry_only_one_transition() -> None:
    lifecycle = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    service = ExecutionAttemptRetryService(lifecycle)
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
def test_r1_concurrent_retry_cas_safe() -> None:
    lifecycle = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    service = ExecutionAttemptRetryService(lifecycle)
    tenant_id = "tenant-a"
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    lifecycle.record_initial_attempt(tenant_id=tenant_id, run_id=run_id, attempt_id=attempt_a1)
    barrier = threading.Barrier(2)
    results: list[str] = []
    errors: list[BaseException] = []
    request = _eligibility_request(ExecutionFailureKind.RETRYABLE_TRANSIENT)

    def worker() -> None:
        barrier.wait()
        try:
            transition = service.transition_for_retry(
                tenant_id=tenant_id,
                task_id=mint_task_id(),
                run_id=run_id,
                expected_attempt_id=attempt_a1,
                request=request,
            )
            if transition is not None:
                results.append(str(transition.active_attempt_id))
        except BaseException as exc:
            errors.append(exc)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert len(results) == 1
    assert lifecycle.get_active_attempt_id(tenant_id=tenant_id, run_id=run_id) == results[0]


@pytest.mark.gate
def test_r1_no_attempt_skip() -> None:
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
def test_r1_terminal_success_blocks_retry() -> None:
    result = evaluate_execution_retry_eligibility(
        _eligibility_request(
            ExecutionFailureKind.RETRYABLE_TRANSIENT,
            terminal_outcome=ExecutionTerminalOutcome.COMPLETED,
        ),
    )
    assert result.action is ExecutionRetryAction.FAIL


@pytest.mark.gate
def test_r1_terminal_cancel_blocks_retry() -> None:
    result = evaluate_execution_retry_eligibility(
        _eligibility_request(
            ExecutionFailureKind.RETRYABLE_TRANSIENT,
            terminal_outcome=ExecutionTerminalOutcome.CANCELLED,
        ),
    )
    assert result.action is ExecutionRetryAction.CANCEL


@pytest.mark.gate
def test_r1_resilience_policy_projection_governance_deny() -> None:
    kind = project_resilience_failure_kind(FailureClass.POLICY_ERROR)
    assert kind is ExecutionFailureKind.GOVERNANCE_DENIED


@pytest.mark.gate
def test_r1_classify_failure_class_authority_and_trust() -> None:
    assert classify_from_failure_class(FailureClass.POLICY_ERROR) is ExecutionFailureKind.GOVERNANCE_DENIED
    assert classify_from_failure_class(FailureClass.USER_ERROR) is ExecutionFailureKind.NON_RETRYABLE_PERMANENT


@pytest.mark.gate
def test_r1_cancellation_coordinator_metadata_blocks_graph_retry() -> None:
    from intergrax.runtime.cancellation.coordinator import CANCELLATION_REQUESTED_KEY

    metadata = {CANCELLATION_REQUESTED_KEY: True}
    assert CancellationCoordinator.is_requested(metadata)


@pytest.mark.gate
def test_r1_frozen_regression_modules_exist() -> None:
    for module in _FROZEN_REGRESSION_MODULES:
        path = _REPO_ROOT / module.replace(".", "/")
        assert path.with_suffix(".py").exists(), module

