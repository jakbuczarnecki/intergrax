# © Artur Czarnecki. All rights reserved.

"""GR-10-R11-R4 — RuntimeToolInvoker exact vs incomplete proposal correlation."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationIdentity,
    ExecutionContinuationLifecycleState,
    PendingExecutionContinuation,
)
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id, validate_run_id
from intergrax.contracts.governed_continuation_correlation import (
    ContinuationReason,
    GovernedContinuationCorrelation,
)
from intergrax.contracts.governed_continuation_grant import GovernedContinuationApprovalGrant
from intergrax.runtime.agent_governance.errors import ToolGovernanceDeniedError
from intergrax.runtime.execution.continuation.persistence import (
    InMemoryExecutionContinuationStateStore,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.tool_invocation_meaningful_side_effect import (
    ORCHESTRATION_TOOL_MSE_OPERATION_ID,
)
from intergrax.runtime.task.task import Task
from testing_support.builder import canonical_run_id_for_tests, canonical_task_id_for_tests
from tests.unit.runtime.nexus.tools.conftest import FakeRegistry
from tests.unit.runtime.nexus.tools.test_gr10_r11_r2_runtime_tool_invoker_post_hitl import (
    _BUNDLE_D,
    _BUNDLE_ID,
    _BUNDLE_V,
    _POLICY_RULE,
    _AllowMseBoundary,
    _CountingExecutor,
    _allow_all_governance,
    _invoke,
    _side_effect_contract,
)
from tests.unit.runtime.nexus.tools.test_gr10_r8_orchestration_inner_guard import (
    _RecordingGuard,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _tool_correlation(
    *,
    continuation_id: str,
    task_id: str,
    run_id: object,
    attempt_id: object,
    execution_id: object,
    side_effect_scope_id: str | None = "probe.side_effect:s1",
    resource_scope: str | None = "probe.side_effect",
) -> GovernedContinuationCorrelation:
    return GovernedContinuationCorrelation(
        continuation_request_id=continuation_id,
        reason=ContinuationReason.COMPLIANCE,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        side_effect_scope_id=side_effect_scope_id,
        operation_id=ORCHESTRATION_TOOL_MSE_OPERATION_ID,
        resource_scope=resource_scope,
        policy_bundle_id=_BUNDLE_ID,
        policy_bundle_version=_BUNDLE_V,
        policy_bundle_digest=_BUNDLE_D,
    )


def test_invoker_exact_tool_proposal_resume_with_grant() -> None:
    executor = _CountingExecutor()
    boundary = _AllowMseBoundary()
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(_side_effect_contract()),
        executor=executor,
        agent_runtime_governance=_allow_all_governance(),
        inner_execution_guard=_RecordingGuard(allow=True),
        meaningful_side_effect_authorization=boundary,
    )
    run_seed = "r11r4-tool-exact"
    run_id = canonical_run_id_for_tests(run_seed)
    task_id = str(canonical_task_id_for_tests(run_seed))
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    store = InMemoryExecutionContinuationStateStore()
    continuation_id = "gcr_r11r4_tool_exact"
    identity = ExecutionContinuationIdentity(
        task_id=task_id,
        run_id=validate_run_id(run_id),
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    store.insert_if_absent(
        PendingExecutionContinuation(
            continuation_id=continuation_id,
            identity=identity,
            lifecycle_state=ExecutionContinuationLifecycleState.RESUMED,
            revision=3,
            reason=ContinuationReason.COMPLIANCE,
            governed_correlation=_tool_correlation(
                continuation_id=continuation_id,
                task_id=task_id,
                run_id=validate_run_id(run_id),
                attempt_id=attempt_id,
                execution_id=execution_id,
            ),
            pause_id="pause-r11r4-exact",
            human_request_id="hr-r11r4-exact",
            requested_at="2026-09-20T00:00:00+00:00",
        )
    )
    grant = GovernedContinuationApprovalGrant.model_validate(
        {
            "grant_id": "gcg_r11r4_tool",
            "continuation_request_id": continuation_id,
            "side_effect_scope_id": "probe.side_effect:s1",
            "task_id": task_id,
            "run_id": validate_run_id(run_id),
            "attempt_id": attempt_id,
            "execution_id": execution_id,
            "operation_id": ORCHESTRATION_TOOL_MSE_OPERATION_ID,
            "resource_scope": "probe.side_effect",
            "policy_rule_id": _POLICY_RULE,
            "policy_bundle_id": _BUNDLE_ID,
            "policy_bundle_version": _BUNDLE_V,
            "policy_bundle_digest": _BUNDLE_D,
            "pause_id": "pause-r11r4-exact",
            "human_request_id": "hr-r11r4-exact",
            "approved_at": "2026-09-20T00:00:00+00:00",
        }
    )
    task = Task(tenant_id="test-tenant", user_id="u1", message="x", task_id=task_id)
    task.runtime.governance.governed_continuation_grant = grant
    _invoke(
        invoker,
        run_seed=run_seed,
        attempt_id=attempt_id,
        execution_id=execution_id,
        continuation_store=store,
        task=task,
    )
    assert boundary.calls == 1
    assert executor.calls == 1
    assert task.runtime.governance.governed_continuation_grant is None


def test_invoker_missing_scope_in_correlation_not_post_hitl() -> None:
    """Correlation missing side_effect_scope_id → not post-HITL; ordinary ALLOW once."""
    executor = _CountingExecutor()
    boundary = _AllowMseBoundary()
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(_side_effect_contract()),
        executor=executor,
        agent_runtime_governance=_allow_all_governance(),
        inner_execution_guard=_RecordingGuard(allow=True),
        meaningful_side_effect_authorization=boundary,
    )
    run_seed = "r11r4-tool-miss-scope"
    run_id = canonical_run_id_for_tests(run_seed)
    task_id = str(canonical_task_id_for_tests(run_seed))
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    store = InMemoryExecutionContinuationStateStore()
    identity = ExecutionContinuationIdentity(
        task_id=task_id,
        run_id=validate_run_id(run_id),
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    store.insert_if_absent(
        PendingExecutionContinuation(
            continuation_id="gcr_r11r4_miss_scope",
            identity=identity,
            lifecycle_state=ExecutionContinuationLifecycleState.RESUMED,
            revision=3,
            reason=ContinuationReason.COMPLIANCE,
            governed_correlation=_tool_correlation(
                continuation_id="gcr_r11r4_miss_scope",
                task_id=task_id,
                run_id=validate_run_id(run_id),
                attempt_id=attempt_id,
                execution_id=execution_id,
                side_effect_scope_id=None,
            ),
            pause_id="pause-miss-scope",
            human_request_id="hr-miss-scope",
            requested_at="2026-09-20T00:00:00+00:00",
        )
    )
    _invoke(
        invoker,
        run_seed=run_seed,
        attempt_id=attempt_id,
        execution_id=execution_id,
        continuation_store=store,
    )
    assert boundary.calls == 1
    assert executor.calls == 1


def test_invoker_exact_match_missing_grant_still_blocks() -> None:
    executor = _CountingExecutor()
    boundary = _AllowMseBoundary()
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(_side_effect_contract()),
        executor=executor,
        agent_runtime_governance=_allow_all_governance(),
        inner_execution_guard=_RecordingGuard(allow=True),
        meaningful_side_effect_authorization=boundary,
    )
    run_seed = "r11r4-tool-no-grant"
    run_id = canonical_run_id_for_tests(run_seed)
    task_id = str(canonical_task_id_for_tests(run_seed))
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    store = InMemoryExecutionContinuationStateStore()
    continuation_id = "gcr_r11r4_no_grant"
    identity = ExecutionContinuationIdentity(
        task_id=task_id,
        run_id=validate_run_id(run_id),
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    store.insert_if_absent(
        PendingExecutionContinuation(
            continuation_id=continuation_id,
            identity=identity,
            lifecycle_state=ExecutionContinuationLifecycleState.RESUMED,
            revision=3,
            reason=ContinuationReason.COMPLIANCE,
            governed_correlation=_tool_correlation(
                continuation_id=continuation_id,
                task_id=task_id,
                run_id=validate_run_id(run_id),
                attempt_id=attempt_id,
                execution_id=execution_id,
            ),
            pause_id="pause-no-grant",
            human_request_id="hr-no-grant",
            requested_at="2026-09-20T00:00:00+00:00",
        )
    )
    with pytest.raises(ToolGovernanceDeniedError):
        _invoke(
            invoker,
            run_seed=run_seed,
            attempt_id=attempt_id,
            execution_id=execution_id,
            continuation_store=store,
        )
    assert boundary.calls == 1
    assert executor.calls == 0
