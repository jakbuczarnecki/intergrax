# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.7-R2 — final sequential authority closure (stale safety + terminal backend)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from intergrax.contracts.collaborative_work import (
    CollaborativeWorkEnforcementRequest,
    CollaborativeWorkEnforcementResult,
    PolicyCompositionResult,
)
from intergrax.contracts.execution.suspended_operation.claim_authority import (
    SuspendedOperationClaimAuthority,
)
from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedOperationMaterializationState,
)
from intergrax.contracts.execution.suspended_operation.reentry import (
    ExecutionSuspendedWorkReentryDisposition,
    ExecutionSuspendedWorkReentryRequest,
)
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.contracts.governed_continuation_correlation import ContinuationReason
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationResult,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.governance.active_governed_execution_task import (
    bind_governed_execution_task,
    reset_governed_execution_task,
)
from intergrax.runtime.human.governed_continuation_bridge import (
    compose_governed_continuation_from_enforcement,
)
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.pause import (
    HumanApprovalResolutionError,
    HumanPauseCoordinator,
)
from intergrax.runtime.task.task import Task, TaskState
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.tool_executor import ToolExecutor
from tests.unit.autonomous_work.test_uca6c_r4_real_codecraft_execution import (
    _TASK_ID,
    _TENANT,
)
from tests.unit.runtime.execution.test_uca6c_r6_r5_7_sequential_authority_generations import (
    _approve_current_pause,
    _build_handler,
    _declarative_policy_bundle,
    _DeclarativeRequireHitlOnceHandler,
    _MseRequireHumanOncePort,
    _start_gen1_pause,
)

pytestmark = pytest.mark.unit


class _MseRequireThenDenyPort:
    def __init__(self) -> None:
        self.calls = 0

    def authorize(
        self,
        request: CollaborativeWorkEnforcementRequest,
        *,
        source_agent_id: str = "platform.orchestration.tool_invocation",
        source_step_id: str | None = None,
    ) -> MeaningfulSideEffectAuthorizationResult:
        self.calls += 1
        action = PolicyAction.REQUIRE_HUMAN if self.calls == 1 else PolicyAction.DENY
        decision = PolicyDecision(
            action=action,
            reason="uca6c-r57-mse-deny",
            policy_rule_id="test.uca6c.r57.mse.deny",
        )
        enforcement_result = CollaborativeWorkEnforcementResult(
            operation_id=request.operation_id,
            authority_scope=request.resource_scope,
            composition=PolicyCompositionResult(
                decision=decision,
                collaborative_authority=decision,
            ),
        )
        requires = action in (PolicyAction.REQUIRE_HUMAN, PolicyAction.ESCALATE)
        governed = compose_governed_continuation_from_enforcement(
            request,
            decision=decision,
            enforcement_operation_id=request.operation_id,
            enforcement_authority_scope=request.resource_scope,
            requires_governed_continuation=requires,
            source_agent_id=source_agent_id,
            source_step_id=source_step_id,
            reason=ContinuationReason.COMPLIANCE,
        )
        return MeaningfulSideEffectAuthorizationResult(
            permitted=action is PolicyAction.ALLOW,
            decision=decision,
            enforcement_result=enforcement_result,
            requires_governed_continuation=requires,
            governed_continuation_request=governed,
        )


class _RaisingToolExecutor:
    def __init__(self, delegate: ToolExecutor) -> None:
        self._delegate = delegate
        self.calls = 0

    def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
        self.calls += 1
        _ = request
        raise RuntimeError("uca6c-r57-backend-fail")


def _task_and_identity() -> tuple[Task, object, object, object, object, object, object]:
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    task = Task(tenant_id=_TENANT, user_id="u1", message="x", task_id=_TASK_ID)
    id_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    gov_token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id=_TENANT,
            workspace_id="workspace-uca6c",
            principal_id="principal-uca6c",
        ),
    )
    task_token = bind_governed_execution_task(task)
    return task, id_token, gov_token, task_token, run_id, attempt_id, execution_id


def _advance_to_gen3_pause(
    tmp_path: Path,
    *,
    mse_port: _MseRequireHumanOncePort | _MseRequireThenDenyPort,
) -> tuple:
    counting = _DeclarativeRequireHitlOnceHandler()
    handler, composition, _, craft_id, hitl, checkpoint_store, backend, _, _ = (
        _build_handler(
            tmp_path,
            mse_port,
            policy_bundle=_declarative_policy_bundle(
                always_require_hitl=False,
                counting_handler=counting,
            ),
        )
    )
    reentry = composition.suspended_work_reentry_coordinator
    store = reentry.store
    task, id_token, gov_token, task_token, run_id, attempt_id, execution_id = (
        _task_and_identity()
    )
    gen1 = _start_gen1_pause(handler, craft_id, execution_id)
    d1 = gen1.descriptor
    fingerprint = d1.logical_invocation_fingerprint
    suspended_id = d1.suspended_operation_id
    c1 = d1.continuation_id
    _approve_current_pause(
        task,
        hitl=hitl,
        continuation_id=c1,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        checkpoint_store=checkpoint_store,
    )
    d2 = store.load_active_for_logical_invocation(fingerprint)
    assert d2 is not None
    c2 = d2.continuation_id
    _approve_current_pause(
        task,
        hitl=hitl,
        continuation_id=c2,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        checkpoint_store=checkpoint_store,
    )
    d3 = store.load_active_for_logical_invocation(fingerprint)
    assert d3 is not None and d3.pause_generation == 3
    return (
        handler,
        composition,
        hitl,
        checkpoint_store,
        backend,
        store,
        task,
        id_token,
        gov_token,
        task_token,
        run_id,
        attempt_id,
        execution_id,
        fingerprint,
        suspended_id,
        c1,
        c2,
        d3,
    )


def test_stale_continuation_reentry_blocks_backend(tmp_path: Path) -> None:
    mse_port = _MseRequireHumanOncePort()
    ctx = _advance_to_gen3_pause(tmp_path, mse_port=mse_port)
    composition = ctx[1]
    backend = ctx[4]
    store = ctx[5]
    id_token = ctx[7]
    gov_token = ctx[8]
    task_token = ctx[9]
    c1 = ctx[15]
    d3 = ctx[17]
    reentry = composition.suspended_work_reentry_coordinator
    try:
        result = reentry.reenter_after_resume(
            ExecutionSuspendedWorkReentryRequest(
                continuation_id=c1,
                identity=d3.identity,
                claim_authority=SuspendedOperationClaimAuthority.for_host_pending_claim(
                    host_owner_id=reentry.claim_owner_id,
                    descriptor=d3,
                ),
            ),
            task=None,
        )
        assert result.disposition in {
            ExecutionSuspendedWorkReentryDisposition.NOT_READY,
            ExecutionSuspendedWorkReentryDisposition.REJECTED,
        }
        assert backend.calls == 0
        active = store.load_active_for_logical_invocation(
            d3.logical_invocation_fingerprint
        )
        assert active is not None
        assert (
            active.materialization_state
            is not SuspendedOperationMaterializationState.CONSUMED
        )
    finally:
        reset_governed_execution_task(task_token)
        reset_active_execution_governance_identity(gov_token)
        reset_active_execution_identity(id_token)


def test_stale_human_request_id_blocks_resume(tmp_path: Path) -> None:
    mse_port = _MseRequireHumanOncePort()
    ctx = _advance_to_gen3_pause(tmp_path, mse_port=mse_port)
    hitl = ctx[2]
    task = ctx[6]
    id_token = ctx[7]
    gov_token = ctx[8]
    task_token = ctx[9]
    run_id = ctx[10]
    attempt_id = ctx[11]
    execution_id = ctx[12]
    try:
        pause_record = task.runtime.governance.pause_record
        human_request = task.runtime.governance.human_request
        assert pause_record is not None and human_request is not None
        approver = local_development_approver_evidence(tenant_id=task.tenant_id)
        with pytest.raises(
            HumanApprovalResolutionError,
            match="human_request_id mismatch",
        ):
            HumanPauseCoordinator.resolve_human_response_and_apply_canonical(
                task,
                HumanResponseVerdict.APPROVE,
                approver=approver,
                continuation=hitl.port,
                pause_id=pause_record.pause_id,
                human_request_id="stale_human_request",
                run_id=str(run_id),
                attempt_id=str(attempt_id),
                execution_id=str(execution_id),
            )
    finally:
        reset_governed_execution_task(task_token)
        reset_active_execution_governance_identity(gov_token)
        reset_active_execution_identity(id_token)


def test_stale_human_pause_id_blocks_resume(tmp_path: Path) -> None:
    mse_port = _MseRequireHumanOncePort()
    ctx = _advance_to_gen3_pause(tmp_path, mse_port=mse_port)
    hitl = ctx[2]
    task = ctx[6]
    id_token = ctx[7]
    gov_token = ctx[8]
    task_token = ctx[9]
    run_id = ctx[10]
    attempt_id = ctx[11]
    execution_id = ctx[12]
    d3 = ctx[17]
    try:
        pause_record = task.runtime.governance.pause_record
        human_request = task.runtime.governance.human_request
        assert pause_record is not None and human_request is not None
        approver = local_development_approver_evidence(tenant_id=task.tenant_id)
        with pytest.raises(HumanApprovalResolutionError, match="pause_id mismatch"):
            HumanPauseCoordinator.resolve_human_response_and_apply_canonical(
                task,
                HumanResponseVerdict.APPROVE,
                approver=approver,
                continuation=hitl.port,
                pause_id="stale_pause_id",
                human_request_id=human_request.request_id,
                run_id=str(run_id),
                attempt_id=str(attempt_id),
                execution_id=str(execution_id),
            )
        assert task.state is TaskState.WAITING_FOR_HUMAN
        assert d3.pause_generation == 3
    finally:
        reset_governed_execution_task(task_token)
        reset_active_execution_governance_identity(gov_token)
        reset_active_execution_identity(id_token)


def test_mse_fresh_deny_after_prior_approvals_blocks_backend(tmp_path: Path) -> None:
    mse_port = _MseRequireThenDenyPort()
    ctx = _advance_to_gen3_pause(tmp_path, mse_port=mse_port)
    hitl = ctx[2]
    checkpoint_store = ctx[3]
    backend = ctx[4]
    store = ctx[5]
    task = ctx[6]
    id_token = ctx[7]
    gov_token = ctx[8]
    task_token = ctx[9]
    run_id = ctx[10]
    attempt_id = ctx[11]
    execution_id = ctx[12]
    fingerprint = ctx[13]
    suspended_id = ctx[14]
    d3 = ctx[17]
    try:
        final = _approve_current_pause(
            task,
            hitl=hitl,
            continuation_id=d3.continuation_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            checkpoint_store=checkpoint_store,
        )
        assert final is not None
        assert (
            final.disposition is not ExecutionSuspendedWorkReentryDisposition.COMPLETED
        )
        assert backend.calls == 0
        terminal = store.load(suspended_id)
        assert terminal is not None
        assert (
            terminal.materialization_state
            is not SuspendedOperationMaterializationState.CONSUMED
        )
        assert store.load_active_for_logical_invocation(fingerprint) is not None
    finally:
        reset_governed_execution_task(task_token)
        reset_active_execution_governance_identity(gov_token)
        reset_active_execution_identity(id_token)


def test_backend_failure_invokes_executor_without_consumed_success(
    tmp_path: Path,
) -> None:
    mse_port = _MseRequireHumanOncePort()
    counting_decl = _DeclarativeRequireHitlOnceHandler()
    handler, composition, _, craft_id, hitl, checkpoint_store, backend, _, _ = (
        _build_handler(
            tmp_path,
            mse_port,
            policy_bundle=_declarative_policy_bundle(
                always_require_hitl=False,
                counting_handler=counting_decl,
            ),
        )
    )
    from intergrax.runtime.nexus.tools.nexus_execution_bound_catalog_tool_invoker import (
        NexusExecutionBoundCatalogToolInvoker,
    )

    invoker = composition.invoker
    assert isinstance(invoker, NexusExecutionBoundCatalogToolInvoker)
    failing = _RaisingToolExecutor(invoker.tool_invoker._executor)  # noqa: SLF001
    invoker.tool_invoker._executor = failing  # noqa: SLF001
    backend = failing

    store = composition.suspended_work_reentry_coordinator.store
    task, id_token, gov_token, task_token, run_id, attempt_id, execution_id = (
        _task_and_identity()
    )
    try:
        gen1 = _start_gen1_pause(handler, craft_id, execution_id)
        c1 = gen1.descriptor.continuation_id
        fingerprint = gen1.descriptor.logical_invocation_fingerprint
        suspended_id = gen1.descriptor.suspended_operation_id
        _approve_current_pause(
            task,
            hitl=hitl,
            continuation_id=c1,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            checkpoint_store=checkpoint_store,
        )
        d2 = store.load_active_for_logical_invocation(fingerprint)
        assert d2 is not None
        _approve_current_pause(
            task,
            hitl=hitl,
            continuation_id=d2.continuation_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            checkpoint_store=checkpoint_store,
        )
        d3 = store.load_active_for_logical_invocation(fingerprint)
        assert d3 is not None
        final = _approve_current_pause(
            task,
            hitl=hitl,
            continuation_id=d3.continuation_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            checkpoint_store=checkpoint_store,
        )
        assert final is not None
        assert final.disposition is ExecutionSuspendedWorkReentryDisposition.FAILED
        assert backend.calls == 1
        terminal = store.load(suspended_id)
        assert terminal is not None
        assert (
            terminal.materialization_state
            is not SuspendedOperationMaterializationState.CONSUMED
        )
    finally:
        reset_governed_execution_task(task_token)
        reset_active_execution_governance_identity(gov_token)
        reset_active_execution_identity(id_token)
