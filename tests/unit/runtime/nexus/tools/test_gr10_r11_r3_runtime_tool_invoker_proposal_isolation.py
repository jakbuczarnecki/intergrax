# © Artur Czarnecki. All rights reserved.

"""GR-10-R11-R3 — RuntimeToolInvoker proposal-scope isolation across tools."""

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
from intergrax.runtime.execution.continuation.persistence import (
    InMemoryExecutionContinuationStateStore,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.tool_invocation_meaningful_side_effect import (
    ORCHESTRATION_TOOL_MSE_OPERATION_ID,
)
from testing_support.builder import canonical_run_id_for_tests, canonical_task_id_for_tests
from tests.unit.runtime.nexus.tools.conftest import FakeRegistry
from tests.unit.runtime.nexus.tools.test_gr10_r11_r2_runtime_tool_invoker_post_hitl import (
    _BUNDLE_D,
    _BUNDLE_ID,
    _BUNDLE_V,
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


def test_invoker_tool_a_hitl_does_not_block_tool_b_same_execution() -> None:
    """Tool A HITL (scope A) resumed; tool B same execution → ordinary ALLOW once."""
    executor = _CountingExecutor()
    boundary = _AllowMseBoundary()
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(_side_effect_contract()),
        executor=executor,
        agent_runtime_governance=_allow_all_governance(),
        inner_execution_guard=_RecordingGuard(allow=True),
        meaningful_side_effect_authorization=boundary,
    )
    run_seed = "r11r3-tool-iso"
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
    # Tool A correlation — different side-effect scope than probe.side_effect:s1.
    store.insert_if_absent(
        PendingExecutionContinuation(
            continuation_id="gcr_r11r3_tool_a",
            identity=identity,
            lifecycle_state=ExecutionContinuationLifecycleState.RESUMED,
            revision=3,
            reason=ContinuationReason.COMPLIANCE,
            governed_correlation=GovernedContinuationCorrelation(
                continuation_request_id="gcr_r11r3_tool_a",
                reason=ContinuationReason.COMPLIANCE,
                task_id=task_id,
                run_id=validate_run_id(run_id),
                attempt_id=attempt_id,
                execution_id=execution_id,
                side_effect_scope_id="other.tool:scope-a",
                side_effect_scope_digest="sha256:" + ("aa" * 32),
                operation_id=ORCHESTRATION_TOOL_MSE_OPERATION_ID,
                resource_scope="other.tool",
                policy_bundle_id=_BUNDLE_ID,
                policy_bundle_version=_BUNDLE_V,
                policy_bundle_digest=_BUNDLE_D,
            ),
            pause_id="pause-tool-a",
            human_request_id="hr-tool-a",
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


def test_invoker_legacy_correlation_less_does_not_block_tool() -> None:
    executor = _CountingExecutor()
    boundary = _AllowMseBoundary()
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(_side_effect_contract()),
        executor=executor,
        agent_runtime_governance=_allow_all_governance(),
        inner_execution_guard=_RecordingGuard(allow=True),
        meaningful_side_effect_authorization=boundary,
    )
    run_seed = "r11r3-tool-legacy"
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
            continuation_id="gcr_r11r3_legacy",
            identity=identity,
            lifecycle_state=ExecutionContinuationLifecycleState.RESUMED,
            revision=2,
            reason=ContinuationReason.COMPLIANCE,
            governed_correlation=None,
            pause_id="pause-legacy",
            human_request_id="hr-legacy-only",
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
