# © Artur Czarnecki. All rights reserved.

"""GR-10-R11-R2 — RuntimeToolInvoker ordinary ALLOW vs post-HITL grant evidence."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
from intergrax.contracts.agent_runtime_governance import CapabilityGrant
from intergrax.contracts.collaborative_work import CollaborativeWorkEnforcementRequest
from intergrax.contracts.execution_continuation import (
    ExecutionContinuationIdentity,
    ExecutionContinuationLifecycleState,
    PendingExecutionContinuation,
)
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    reset_active_execution_identity,
    validate_run_id,
)
from intergrax.contracts.governed_continuation_correlation import (
    ContinuationReason,
    GovernedContinuationCorrelation,
)
from intergrax.contracts.governed_continuation_grant import GovernedContinuationApprovalGrant
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationResult,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.agent_governance.audit import (
    GovernanceAuditRecorder,
    InMemoryGovernanceAuditSink,
)
from intergrax.runtime.agent_governance.authorization_boundary import (
    AgentRuntimeGovernanceBoundary,
)
from intergrax.runtime.agent_governance.capability_resolver import (
    InMemoryCapabilityGrantResolver,
)
from intergrax.runtime.agent_governance.errors import ToolGovernanceDeniedError
from intergrax.runtime.agent_governance.pipeline import AgentRuntimeGovernancePipeline
from intergrax.runtime.agent_governance.policy_engine import (
    AgentRuntimePolicyEngine,
    AllowAllPolicyProvider,
)
from intergrax.runtime.execution.active_execution_continuation_store import (
    bind_active_execution_continuation_state_store,
    reset_active_execution_continuation_state_store,
)
from intergrax.runtime.execution.continuation.persistence import (
    InMemoryExecutionContinuationStateStore,
)
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.governance.active_governed_execution_task import (
    ActiveGovernedExecutionTask,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.tool_invocation_meaningful_side_effect import (
    ORCHESTRATION_TOOL_MSE_OPERATION_ID,
)
from intergrax.runtime.task.task import Task
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.execution_models import ToolExecutionRequest
from testing_support.builder import (
    build_runtime_state_for_tests,
    canonical_run_id_for_tests,
    canonical_task_id_for_tests,
)
from tests.unit.runtime.nexus.tools.conftest import FakeRegistry
from tests.unit.runtime.nexus.tools.test_gr10_r8_orchestration_inner_guard import (
    _RecordingGuard,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_BUNDLE_ID = "bundle-r11r2-tool"
_BUNDLE_V = "1.0.0"
_BUNDLE_D = "sha256:" + ("55" * 32)
_POLICY_RULE = "test.mse.r11r2"


class _In(BaseModel):
    value: int


class _Out(BaseModel):
    value: int


class _CountingExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
        self.calls += 1
        return _Out(value=request.input.value)


class _AllowMseBoundary:
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
        decision = PolicyDecision(
            action=PolicyAction.ALLOW,
            reason="r11r2-tool-allow",
            policy_rule_id=_POLICY_RULE,
            policy_bundle_id=_BUNDLE_ID,
            policy_bundle_version=_BUNDLE_V,
            policy_bundle_digest=_BUNDLE_D,
        )
        from intergrax.contracts.collaborative_work import (
            CollaborativeWorkEnforcementResult,
            PolicyCompositionResult,
        )

        enforcement_result = CollaborativeWorkEnforcementResult(
            operation_id=request.operation_id,
            authority_scope=request.resource_scope,
            composition=PolicyCompositionResult(
                decision=decision,
                collaborative_authority=decision,
            ),
        )
        return MeaningfulSideEffectAuthorizationResult(
            permitted=True,
            decision=decision,
            enforcement_result=enforcement_result,
            requires_governed_continuation=False,
            governed_continuation_request=None,
        )


def _side_effect_contract() -> ToolContract:
    return ToolContract(
        tool_id="probe.side_effect",
        name="probe",
        description="probe",
        input_schema=_In,
        output_schema=_Out,
        side_effects=True,
        error_mapping={},
        risk_level=ToolRiskLevel.HIGH,
    )


def _enforce_allow_policy_bundle() -> object:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="gr10.r11r2.allow")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[],
        policy_enforcement_mode="enforce",
    )
    return wire_policy_bundle(env)


def _allow_all_governance() -> AgentRuntimeGovernanceBoundary:
    grant = CapabilityGrant(
        tenant_id="test-tenant",
        agent_id="agent-1",
        allowed_capabilities=frozenset({"probe.side_effect"}),
    )
    pipeline = AgentRuntimeGovernancePipeline(
        capability_resolver=InMemoryCapabilityGrantResolver((grant,)),
        policy_engine=AgentRuntimePolicyEngine((AllowAllPolicyProvider(),)),
        audit_recorder=GovernanceAuditRecorder(InMemoryGovernanceAuditSink()),
    )
    return AgentRuntimeGovernanceBoundary(pipeline)


def _invoke(
    invoker: RuntimeToolInvoker,
    *,
    run_seed: str,
    attempt_id: str,
    execution_id: str,
    continuation_store: InMemoryExecutionContinuationStateStore | None = None,
    task: Task | None = None,
) -> tuple[str, str]:
    run_id = canonical_run_id_for_tests(run_seed)
    task_id = str(canonical_task_id_for_tests(run_seed))
    state = build_runtime_state_for_tests(run_id=run_seed)
    state.context.config.production_mode = True
    state.context.config.policy_bundle = _enforce_allow_policy_bundle()
    request = ToolExecutionRequest(
        run_id=validate_run_id(run_id),
        tool_id="probe.side_effect",
        step_id="s1",
        input=_In(value=1),
    )
    identity_token = bind_active_execution_identity(
        run_id=validate_run_id(run_id),
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    gov_token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id="test-tenant",
            workspace_id="test-workspace",
            principal_id="principal-A",
        ),
    )
    cont_token = None
    if continuation_store is not None:
        cont_token = bind_active_execution_continuation_state_store(continuation_store)
    governed = ActiveGovernedExecutionTask()
    gov_task_token = None
    if task is not None:
        gov_task_token = governed.bind(task)
    try:
        invoker.invoke(state=state, agent_id="agent-1", request=request)
    finally:
        if gov_task_token is not None:
            governed.reset(gov_task_token)
        if cont_token is not None:
            reset_active_execution_continuation_state_store(cont_token)
        reset_active_execution_governance_identity(gov_token)
        reset_active_execution_identity(identity_token)
    return task_id, run_id


def test_invoker_ordinary_allow_without_grant_provider_once() -> None:
    executor = _CountingExecutor()
    boundary = _AllowMseBoundary()
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(_side_effect_contract()),
        executor=executor,
        agent_runtime_governance=_allow_all_governance(),
        inner_execution_guard=_RecordingGuard(allow=True),
        meaningful_side_effect_authorization=boundary,
    )
    _invoke(
        invoker,
        run_seed="r11r2-ord",
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    assert boundary.calls == 1
    assert executor.calls == 1


def test_invoker_post_hitl_resumed_missing_grant_zero_provider_calls() -> None:
    executor = _CountingExecutor()
    boundary = _AllowMseBoundary()
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(_side_effect_contract()),
        executor=executor,
        agent_runtime_governance=_allow_all_governance(),
        inner_execution_guard=_RecordingGuard(allow=True),
        meaningful_side_effect_authorization=boundary,
    )
    run_seed = "r11r2-hitl-miss"
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
    continuation_id = "gcr_r11r2_tool"
    correlation = GovernedContinuationCorrelation(
        continuation_request_id=continuation_id,
        reason=ContinuationReason.COMPLIANCE,
        task_id=task_id,
        run_id=validate_run_id(run_id),
        attempt_id=attempt_id,
        execution_id=execution_id,
        side_effect_scope_id="probe.side_effect:s1",
        operation_id=ORCHESTRATION_TOOL_MSE_OPERATION_ID,
        resource_scope="probe.side_effect",
        policy_bundle_id=_BUNDLE_ID,
        policy_bundle_version=_BUNDLE_V,
        policy_bundle_digest=_BUNDLE_D,
    )
    store.insert_if_absent(
        PendingExecutionContinuation(
            continuation_id=continuation_id,
            identity=identity,
            lifecycle_state=ExecutionContinuationLifecycleState.RESUMED,
            revision=3,
            reason=ContinuationReason.COMPLIANCE,
            governed_correlation=correlation,
            pause_id="pause-tool",
            human_request_id="hr-tool",
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


def test_invoker_post_hitl_resumed_matching_grant_provider_once() -> None:
    executor = _CountingExecutor()
    boundary = _AllowMseBoundary()
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(_side_effect_contract()),
        executor=executor,
        agent_runtime_governance=_allow_all_governance(),
        inner_execution_guard=_RecordingGuard(allow=True),
        meaningful_side_effect_authorization=boundary,
    )
    run_seed = "r11r2-hitl-ok"
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
    continuation_id = "gcr_r11r2_tool_ok"
    correlation = GovernedContinuationCorrelation(
        continuation_request_id=continuation_id,
        reason=ContinuationReason.COMPLIANCE,
        task_id=task_id,
        run_id=validate_run_id(run_id),
        attempt_id=attempt_id,
        execution_id=execution_id,
        side_effect_scope_id="probe.side_effect:s1",
        operation_id=ORCHESTRATION_TOOL_MSE_OPERATION_ID,
        resource_scope="probe.side_effect",
        policy_bundle_id=_BUNDLE_ID,
        policy_bundle_version=_BUNDLE_V,
        policy_bundle_digest=_BUNDLE_D,
    )
    store.insert_if_absent(
        PendingExecutionContinuation(
            continuation_id=continuation_id,
            identity=identity,
            lifecycle_state=ExecutionContinuationLifecycleState.RESUMED,
            revision=3,
            reason=ContinuationReason.COMPLIANCE,
            governed_correlation=correlation,
            pause_id="pause-tool-ok",
            human_request_id="hr-tool-ok",
            requested_at="2026-09-20T00:00:00+00:00",
        )
    )
    task = Task(tenant_id="test-tenant", user_id="u1", message="x", task_id=task_id)
    task.runtime.governance.governed_continuation_grant = (
        GovernedContinuationApprovalGrant.model_validate(
            {
                "grant_id": "gcg_r11r2_tool",
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
                "pause_id": "pause-tool-ok",
                "human_request_id": "hr-tool-ok",
                "approved_at": "2026-09-20T00:00:00+00:00",
            }
        )
    )
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
