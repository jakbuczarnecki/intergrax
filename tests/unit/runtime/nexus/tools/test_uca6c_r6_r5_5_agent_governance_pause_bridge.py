# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.5 — Agent Governance EE L3 canonical pause bridge."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.applications._shared.uca6c_codecraft_qualified_execution_composition import (
    build_production_codecraft_qualified_capability_execution_handler,
)
from intergrax.contracts.execution.suspended_operation.authority_scope import (
    SuspendedOperationAuthorityScope,
)
from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedOperationMaterializationState,
)
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.contracts.governed_continuation import GovernedContinuationRequest
from intergrax.runtime.agent_governance.errors import (
    ToolGovernanceApprovalRequiredError,
)
from intergrax.runtime.execution.suspended_operation.pause_required import (
    ExecutionSuspendedWorkPauseRequired,
)
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.governance.active_governed_execution_task import (
    bind_governed_execution_task,
    reset_governed_execution_task,
)
from intergrax.runtime.nexus.tools.agent_governance_approval_pause_bridge import (
    translate_agent_governance_approval_error,
)
from intergrax.runtime.task.task import Task, TaskState
from intergrax.contracts.codecraft.bound_capability_execution import (
    CodeCraftBoundCapabilityExecutionRequest,
)
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationPort,
    MeaningfulSideEffectAuthorizationResult,
)
from intergrax.contracts.collaborative_work import (
    CollaborativeWorkEnforcementResult,
    PolicyCompositionResult,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.contracts.agent_runtime_governance import (
    AgentIdentity,
    ToolAuthorizationRequest,
    ToolAuthorizationRiskLevel,
)
from intergrax.contracts.governed_continuation_correlation import ContinuationReason
from tests.unit.autonomous_work.test_uca6c_r4_real_codecraft_execution import (
    _TASK_ID,
    _TENANT,
)
from tests.unit.autonomous_work.test_uca6c_r5_r2_strict_governance_composition import (
    _RecordingMsePort,
    _codecraft_context,
    _strict_r6_kwargs,
    _strict_tool_wiring,
)
from tests.unit.autonomous_work.uca6c_r5_r2_strict_fixtures import (
    uca6c_strict_r6_durable_wiring,
    uca6c_strict_sandbox_env_profile,
    uca6c_strict_worker_manifest,
    uca6c_strict_worker_registry,
)
from tests.unit.runtime.nexus.tools.test_gr10_r8_orchestration_inner_guard import (
    _RecordingGuard,
)

pytestmark = pytest.mark.unit


def test_mse_governed_continuation_error_is_not_agent_bridge_input() -> None:
    from intergrax.contracts.execution_identity import mint_task_id

    task_id = mint_task_id()
    run_id = mint_run_id()
    auth = ToolAuthorizationRequest(
        agent=AgentIdentity(agent_id="agent-a", tenant_id="tenant-a"),
        task_id=task_id,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        capability="sandbox",
        tool_id="code.exec",
        requested_action="execute:code.exec",
        risk_level=ToolAuthorizationRiskLevel.HIGH,
    )
    error = ToolGovernanceApprovalRequiredError(
        run_id=str(run_id),
        agent_id="agent-a",
        tool_id="code.exec",
        capability="sandbox",
        approval_id="mse",
        reason="mse",
        policy_results=(),
        governed_continuation_request=GovernedContinuationRequest(
            reason=ContinuationReason.COMPLIANCE,
            task_id=task_id,
            run_id=run_id,
            attempt_id=auth.attempt_id,
            execution_id=auth.execution_id,
            source_agent_id="agent-a",
            prompt="mse pause",
            operation_id="mse_op",
        ),
    )
    with pytest.raises(ToolGovernanceApprovalRequiredError):
        translate_agent_governance_approval_error(
            error,
            authorization_request=auth,
            execution_id=str(auth.execution_id),
            step_id="step-1",
        )


def test_agent_governance_first_pause_materializes_waiting_task(tmp_path: Path) -> None:
    craft_id = "craft-r5-5-agent-pause"
    bundle = uca6c_strict_r6_durable_wiring(tmp_path)
    ctx = _codecraft_context(
        tmp_path,
        craft_id,
        sandbox_manager=bundle["sandbox_session_manager"],
    )
    manifest = uca6c_strict_worker_manifest()
    registry = uca6c_strict_worker_registry(manifest)
    tool_wiring = _strict_tool_wiring(ctx)
    r6_kwargs = _strict_r6_kwargs(tmp_path)
    handler = build_production_codecraft_qualified_capability_execution_handler(
        tool_wiring,
        uca6c_strict_sandbox_env_profile(),
        caller_agent_id="worker-uca6c-qualified",
        tenant_id=_TENANT,
        manifest=manifest,
        agent_registry=registry,
        meaningful_side_effect_authorization=_RecordingMsePort(allow=True),
        canonical_inner_execution_guard=_RecordingGuard(allow=True),
        document_store=r6_kwargs["document_store"],
        continuation_dependencies=r6_kwargs["continuation_dependencies"],
        durable_wiring_binding_resolver=r6_kwargs.get(
            "durable_wiring_binding_resolver"
        ),
    )
    port = handler._execution_port
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
    try:
        with pytest.raises(ExecutionSuspendedWorkPauseRequired) as exc_info:
            port.execute(
                CodeCraftBoundCapabilityExecutionRequest(
                    craft_id=craft_id,
                    tenant_id=_TENANT,
                    task_id=_TASK_ID,
                    run_id=None,
                    execution_id=execution_id,
                    execution_request_id="uca6c-r5-5-agent-pause",
                ),
            )
    finally:
        reset_governed_execution_task(task_token)
        reset_active_execution_governance_identity(gov_token)
        reset_active_execution_identity(id_token)

    pause_exc = exc_info.value
    assert pause_exc.agent_governance_pause is not None
    descriptor = pause_exc.descriptor
    assert (
        descriptor.materialization_state
        is SuspendedOperationMaterializationState.BLOCKED
    )
    assert descriptor.pause_generation == 1
    assert (
        descriptor.authority_scope
        is SuspendedOperationAuthorityScope.AGENT_RUNTIME_GOVERNANCE
    )
    assert descriptor.invocation_scope_id.startswith("agr_")
    assert pause_exc.governed_request.operation_id == descriptor.invocation_scope_id
    assert str(task.task_id) == str(_TASK_ID)
    assert task.state is TaskState.WAITING_FOR_HUMAN
    pending = task.runtime.governance.agent_governance_hitl_pending
    assert pending is not None
    assert pending.generation == 1
    assert (
        pending.agent_governance_invocation_scope_id == descriptor.invocation_scope_id
    )
    assert task.runtime.governance.human_request is not None
    assert task.runtime.governance.agent_governance_human_approval_grant is None
    assert descriptor.identity.task_id == task.task_id
    assert str(descriptor.identity.run_id) == str(run_id)
    assert str(descriptor.identity.attempt_id) == str(attempt_id)
    assert str(descriptor.identity.execution_id) == str(execution_id)
