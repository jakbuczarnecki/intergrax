# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.7 — Agent → Declarative → MSE sequential authority generations."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import BaseModel

from intergrax.applications._shared.uca6c_codecraft_qualified_execution_composition import (
    bootstrap_uca6c_code_exec_catalog_tools,
)
from intergrax.runtime.codecraft.qualified_capability_execution_wiring import (
    build_codecraft_qualified_capability_execution_handler,
)
from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.runtime.execution.execution_bound_catalog_tool_composition import (
    build_execution_bound_catalog_tool_composition,
)
from intergrax.runtime.wiring.agent_runtime_governance_factory import (
    build_agent_runtime_governance_boundary,
)
from intergrax.applications._shared.agent_runtime_governance_wiring import (
    capability_grants_from_application_manifest,
)
from intergrax.runtime.tools.scope_policy import StaticToolScopePolicy
from intergrax.runtime.sandbox.isolation_gate import sandbox_availability_provider
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
)
from intergrax.applications.contracts.environment_profile.sub_profiles import (
    PolicyRulesProfile,
)
from intergrax.contracts.codecraft.bound_capability_execution import (
    CodeCraftBoundCapabilityExecutionRequest,
)
from intergrax.contracts.collaborative_work import (
    CollaborativeWorkEnforcementRequest,
    CollaborativeWorkEnforcementResult,
    PolicyCompositionResult,
)
from intergrax.contracts.execution_continuation import ExecutionContinuationLookup
from intergrax.contracts.execution.suspended_operation.authority_scope import (
    SuspendedOperationAuthorityScope,
)
from intergrax.contracts.execution.suspended_operation.reentry import (
    ExecutionSuspendedWorkReentryDisposition,
)
from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedOperationMaterializationState,
)
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.tool_executor import ToolExecutor
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
    MeaningfulSideEffectAuthorizationPort,
    MeaningfulSideEffectAuthorizationResult,
)
from intergrax.contracts.orchestration_tool_invocation_mse_operation import (
    CANONICAL_ORCHESTRATION_TOOL_MSE_OPERATION_ID,
)
from intergrax.contracts.policy_enforcement_mode import PolicyEnforcementMode
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.core.plugins.admission import DomainPluginLoadReport
from intergrax.core.plugins.discovery import EP_POLICY_RULES
from intergrax.runtime.policy.policy_bundle import (
    DeclarativePolicyRuntime,
    RuntimePolicyBundle,
)
from intergrax.runtime.policy.rules.evaluation import (
    PolicyEnforcementMode as DeclarativeEnforcementMode,
)
from intergrax.runtime.policy.rules.provenance import PolicyBundleProvenance
from intergrax.runtime.policy.rules.registry import PolicyRuleRegistry
from intergrax.runtime.policy.rules.schema import (
    DeclarativePolicyRule,
    PolicyRuleAction,
)
from intergrax.runtime.execution.suspended_operation.authorized_resume_reentry import (
    resume_authorized_continuation_with_suspended_work_reentry,
)
from intergrax.runtime.nexus.orchestration.internal_continuation_orchestration import (
    InternalOrchestrationContinuation,
)
from intergrax.runtime.execution.suspended_operation.pause_required import (
    ExecutionSuspendedWorkPauseRequired,
)
from intergrax.runtime.human.agent_governance_human_approval_grant import (
    AgentGovernanceHumanApprovalGrantCoordinator,
)
from intergrax.runtime.human.declarative_hitl_grant import (
    DeclarativeHitlGrantCoordinator,
)
from intergrax.runtime.human.governed_continuation_bridge import (
    compose_governed_continuation_from_enforcement,
)
from intergrax.runtime.human.governed_continuation_grant import (
    GovernedContinuationGrantCoordinator,
)
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.governance.active_governed_execution_task import (
    bind_governed_execution_task,
    reset_governed_execution_task,
)
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.task.task import Task, TaskState
from intergrax.tools.providers.sandbox.bundle import CODE_EXEC_TOOL_ID
from tests.unit.autonomous_work.test_uca6c_r4_real_codecraft_execution import (
    _TASK_ID,
    _TENANT,
)
from tests.unit.autonomous_work.test_uca6c_r5_r2_strict_governance_composition import (
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

_DECLARATIVE_RULE_ID = "uca6c.r57.code_exec_hitl"
_COUNTING_HANDLER_ID = "uca6c_r57_counting_hitl"


class _CountingToolExecutor:
    """Delegates to the canonical registry executor; counts physical backend invocations."""

    def __init__(self, delegate: ToolExecutor) -> None:
        self._delegate = delegate
        self.calls = 0

    def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
        self.calls += 1
        return self._delegate.execute(request)


_EMPTY_POLICY_PROVENANCE = PolicyBundleProvenance(
    source_kind="inline",
    rules_path=None,
    rules_digest_sha256="uca6c-r57-test",
    handler_provenance=(),
)


class _DeclarativeRequireHitlOnceHandler:
    """Test policy: REQUIRE_HITL on first matching evaluation, then ALLOW."""

    rule_id = _COUNTING_HANDLER_ID

    def __init__(self) -> None:
        self.matching_evaluations = 0

    def evaluate(
        self,
        rule: DeclarativePolicyRule,
        *,
        context: object,
    ) -> PolicyRuleAction:
        from intergrax.runtime.policy.rules.evaluation import PolicyEvaluationContext

        if not isinstance(context, PolicyEvaluationContext):
            return PolicyRuleAction.ALLOW
        if rule.resource_kind != "tool":
            return PolicyRuleAction.ALLOW
        if rule.resource_id != "*" and rule.resource_id != context.tool_id:
            return PolicyRuleAction.ALLOW
        self.matching_evaluations += 1
        if self.matching_evaluations <= 1:
            return PolicyRuleAction.REQUIRE_HITL
        return PolicyRuleAction.ALLOW


def _declarative_policy_bundle(
    *,
    always_require_hitl: bool,
    counting_handler: _DeclarativeRequireHitlOnceHandler | None = None,
) -> RuntimePolicyBundle:
    env = (
        _r57_env_profile()
        if always_require_hitl
        else uca6c_strict_sandbox_env_profile()
    )
    bundle = wire_policy_bundle(env)
    if always_require_hitl:
        return bundle
    handler = counting_handler or _DeclarativeRequireHitlOnceHandler()
    registry = PolicyRuleRegistry()
    registry.register(handler)
    rule = DeclarativePolicyRule(
        rule_id=_DECLARATIVE_RULE_ID,
        handler_id=handler.rule_id,
        resource_kind="tool",
        resource_id=CODE_EXEC_TOOL_ID,
        action=PolicyRuleAction.REQUIRE_HITL,
    )
    runtime = DeclarativePolicyRuntime(
        registry=registry,
        rules=(rule,),
        load_report=DomainPluginLoadReport.empty(EP_POLICY_RULES),
        provenance=_EMPTY_POLICY_PROVENANCE,
        enforcement_mode=DeclarativeEnforcementMode.ENFORCE,
    )
    return replace(bundle, declarative_policy_runtime=runtime)


def _r57_env_profile() -> ApplicationEnvironmentProfile:
    base = uca6c_strict_sandbox_env_profile()
    return base.model_copy(
        update={
            "policy_rules": PolicyRulesProfile(
                inline_rules=[
                    {
                        "rule_id": _DECLARATIVE_RULE_ID,
                        "handler_id": "deny_tool",
                        "resource_kind": "tool",
                        "resource_id": CODE_EXEC_TOOL_ID,
                        "action": "require_hitl",
                    },
                ],
                policy_enforcement_mode=PolicyEnforcementMode.ENFORCE,
            ),
        },
    )


class _MseRequireHumanOncePort:
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
        action = PolicyAction.REQUIRE_HUMAN if self.calls == 1 else PolicyAction.ALLOW
        decision = PolicyDecision(
            action=action,
            reason="uca6c-r57-mse",
            policy_rule_id="test.uca6c.r57.mse",
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


def _build_handler(
    tmp_path: Path,
    mse_port: MeaningfulSideEffectAuthorizationPort,
    *,
    policy_bundle: RuntimePolicyBundle | None = None,
    idempotency_store: InMemoryIdempotencyStore | None = None,
    inner_guard: _RecordingGuard | None = None,
    tool_executor: ToolExecutor | None = None,
):
    craft_id = "craft-r5-7-sequential"
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
    side_effects: list[str] = []
    env = _r57_env_profile()
    bootstrap_uca6c_code_exec_catalog_tools(tool_wiring)
    grants = capability_grants_from_application_manifest(
        manifest,
        tenant_id=_TENANT,
        agent_registry=registry,
    )
    guard = inner_guard or _RecordingGuard(allow=True)
    resolved_policy = (
        policy_bundle if policy_bundle is not None else wire_policy_bundle(env)
    )
    idem = idempotency_store or InMemoryIdempotencyStore()
    base_executor = RegistryToolExecutor(registry=tool_wiring.registry)
    counting_executor = (
        tool_executor
        if tool_executor is not None
        else _CountingToolExecutor(base_executor)
    )
    if not isinstance(counting_executor, _CountingToolExecutor):
        counting_executor = _CountingToolExecutor(counting_executor)
    composition = build_execution_bound_catalog_tool_composition(
        registry=tool_wiring.registry,
        policy_bundle=resolved_policy,
        caller_agent_id="worker-uca6c-qualified",
        sandbox_availability=sandbox_availability_provider(tool_wiring.wiring_context),
        production_mode=True,
        scope_policy=StaticToolScopePolicy(allowed_tools={CODE_EXEC_TOOL_ID}),
        agent_runtime_governance=build_agent_runtime_governance_boundary(
            capability_grants=grants,
        ),
        canonical_inner_execution_guard=guard,
        meaningful_side_effect_authorization=mse_port,
        document_store=r6_kwargs["document_store"],
        continuation_dependencies=r6_kwargs["continuation_dependencies"],
        reentry_claim_owner_id="uca6c:worker-uca6c-qualified",
        durable_wiring_binding_resolver=r6_kwargs.get(
            "durable_wiring_binding_resolver"
        ),
        task_checkpoint_store=r6_kwargs["task_checkpoint_store"],
        idempotency_store=idem,
        tool_executor=counting_executor,
    )
    handler = build_codecraft_qualified_capability_execution_handler(
        tool_wiring.wiring_context,
        catalog_tool_invoker=composition.invoker,
        side_effect_recorder=side_effects,
    )
    hitl = InternalOrchestrationContinuation(
        port=r6_kwargs["continuation_dependencies"].continuation,
        lifecycle_driver=r6_kwargs["continuation_dependencies"].lifecycle_driver,
        suspended_work_reentry_coordinator=composition.suspended_work_reentry_coordinator,
    )
    return (
        handler,
        composition,
        side_effects,
        craft_id,
        hitl,
        r6_kwargs["task_checkpoint_store"],
        counting_executor,
        guard,
    )


def _approve_current_pause(
    task: Task,
    *,
    hitl,
    continuation_id: str,
    run_id,
    attempt_id,
    execution_id,
    checkpoint_store,
) -> None:
    pending = hitl.port.get_pending(
        ExecutionContinuationLookup(continuation_id=continuation_id),
    )
    pause_record = task.runtime.governance.pause_record
    human_request = task.runtime.governance.human_request
    assert pause_record is not None and human_request is not None
    if pending.governed_correlation is not None:
        task.runtime.governance.human_request = human_request.model_copy(
            update={"governed_continuation": pending.governed_correlation},
        )
    approver = local_development_approver_evidence(tenant_id=task.tenant_id)
    authorized = HumanPauseCoordinator.resolve_human_response_and_apply_canonical(
        task,
        HumanResponseVerdict.APPROVE,
        approver=approver,
        continuation=hitl.port,
        pause_id=pause_record.pause_id,
        human_request_id=human_request.request_id,
        run_id=str(run_id),
        attempt_id=str(attempt_id),
        execution_id=str(execution_id),
    )
    declarative_pending = task.runtime.governance.declarative_hitl_pending
    if (
        declarative_pending is not None
        and declarative_pending.pause_id == pause_record.pause_id
    ):
        DeclarativeHitlGrantCoordinator.create_grant_from_pending(task)
    agent_pending = task.runtime.governance.agent_governance_hitl_pending
    if (
        agent_pending is not None
        and checkpoint_store is not None
        and agent_pending.pause_id == pause_record.pause_id
    ):
        AgentGovernanceHumanApprovalGrantCoordinator.persist_available_grant_from_human_approve(
            task,
            checkpoint_store=checkpoint_store,
            approver=approver,
        )
    if task.runtime.governance.human_request is not None:
        GovernedContinuationGrantCoordinator.create_grant_from_approval(task)
    _resumed, reentry_result = (
        resume_authorized_continuation_with_suspended_work_reentry(
            task,
            authorized,
            capability=hitl,
            reentry_coordinator=hitl.suspended_work_reentry_coordinator,
        )
    )
    if (
        reentry_result is not None
        and reentry_result.disposition
        is ExecutionSuspendedWorkReentryDisposition.PAUSED_FOR_NEXT_AUTHORITY
    ):
        task.state = TaskState.WAITING_FOR_HUMAN
        task.sync_metadata()
    return reentry_result


def _start_gen1_pause(handler, craft_id: str, execution_id) -> object:
    with pytest.raises(ExecutionSuspendedWorkPauseRequired) as exc:
        handler._execution_port.execute(  # noqa: SLF001
            CodeCraftBoundCapabilityExecutionRequest(
                craft_id=craft_id,
                tenant_id=_TENANT,
                task_id=_TASK_ID,
                run_id=None,
                execution_id=execution_id,
                execution_request_id="uca6c-r57-gen1",
            ),
        )
    return exc.value


def _assert_four_ids_match(descriptor, *, run_id, fingerprint) -> None:
    assert str(descriptor.identity.task_id) == str(_TASK_ID)
    assert str(descriptor.identity.run_id) == str(run_id)
    assert descriptor.logical_invocation_fingerprint == fingerprint


def test_three_authority_generations_without_backend_when_declarative_always_requires(
    tmp_path: Path,
) -> None:
    mse_port = _MseRequireHumanOncePort()
    handler, composition, side_effects, craft_id, hitl, checkpoint_store, _, _ = (
        _build_handler(
            tmp_path,
            mse_port,
        )
    )
    reentry = composition.suspended_work_reentry_coordinator
    assert reentry is not None
    store = reentry.store

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
        gen1 = _start_gen1_pause(handler, craft_id, execution_id)
        assert gen1.agent_governance_pause is not None
        d1 = gen1.descriptor
        assert d1.pause_generation == 1
        assert (
            d1.authority_scope
            is SuspendedOperationAuthorityScope.AGENT_RUNTIME_GOVERNANCE
        )
        assert d1.invocation_scope_id.startswith("agr_")
        suspended_id = d1.suspended_operation_id
        fingerprint = d1.logical_invocation_fingerprint
        c1 = d1.continuation_id
        assert side_effects == []

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
        assert d2.suspended_operation_id == suspended_id
        assert d2.pause_generation == 2
        assert (
            d2.authority_scope
            is SuspendedOperationAuthorityScope.DECLARATIVE_GOVERNANCE
        )
        assert d2.invocation_scope_id.startswith("dhr_")
        assert d2.continuation_id != c1
        assert side_effects == []

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
        assert d3 is not None
        assert d3.pause_generation == 3
        assert (
            d3.authority_scope
            is SuspendedOperationAuthorityScope.MEANINGFUL_SIDE_EFFECT
        )
        assert (
            d3.invocation_scope_id == CANONICAL_ORCHESTRATION_TOOL_MSE_OPERATION_ID
            or d3.invocation_scope_id.startswith(
                f"{CANONICAL_ORCHESTRATION_TOOL_MSE_OPERATION_ID}:",
            )
        )
        assert side_effects == []

        assert mse_port.calls == 1
        c3 = d3.continuation_id
        _approve_current_pause(
            task,
            hitl=hitl,
            continuation_id=c3,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            checkpoint_store=checkpoint_store,
        )
        d4 = store.load_active_for_logical_invocation(fingerprint)
        assert d4 is not None
        assert d4.pause_generation == 4
        assert (
            d4.authority_scope
            is SuspendedOperationAuthorityScope.DECLARATIVE_GOVERNANCE
        )
        assert d4.invocation_scope_id.startswith("dhr_")
        assert d4.suspended_operation_id == suspended_id
        assert d4.continuation_id not in {c1, c2, c3}
        assert side_effects == []
        assert task.state is TaskState.WAITING_FOR_HUMAN
    finally:
        reset_governed_execution_task(task_token)
        reset_active_execution_governance_identity(gov_token)
        reset_active_execution_identity(id_token)

    for gen in (d1, d2, d3, d4):
        _assert_four_ids_match(gen, run_id=run_id, fingerprint=fingerprint)
        assert str(gen.identity.attempt_id) == str(attempt_id)
        assert str(gen.identity.execution_id) == str(execution_id)

    pending_c1 = hitl.port.get_pending(ExecutionContinuationLookup(continuation_id=c1))
    assert pending_c1.lifecycle_state.value in {"resumed", "cancelled", "rejected"}


def test_eventual_backend_when_each_authority_requires_human_once(
    tmp_path: Path,
) -> None:
    counting = _DeclarativeRequireHitlOnceHandler()
    mse_port = _MseRequireHumanOncePort()
    (
        handler,
        composition,
        side_effects,
        craft_id,
        hitl,
        checkpoint_store,
        backend,
        guard,
    ) = _build_handler(
        tmp_path,
        mse_port,
        policy_bundle=_declarative_policy_bundle(
            always_require_hitl=False,
            counting_handler=counting,
        ),
    )
    store = composition.suspended_work_reentry_coordinator.store
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
        assert d2 is not None and d2.pause_generation == 2
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
        c3 = d3.continuation_id

        final_reentry = _approve_current_pause(
            task,
            hitl=hitl,
            continuation_id=c3,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            checkpoint_store=checkpoint_store,
        )
        assert final_reentry is not None
        assert (
            final_reentry.disposition
            is ExecutionSuspendedWorkReentryDisposition.COMPLETED
        )
        assert final_reentry.tool_result is not None
        assert final_reentry.tool_result.success is True
        assert side_effects == []
        assert backend.calls == 1
        assert guard.calls >= 1
        assert mse_port.calls == 2
        assert counting.matching_evaluations == 3
        active = store.load_active_for_logical_invocation(fingerprint)
        assert active is None
        terminal = store.load(suspended_id)
        assert terminal is not None
        assert (
            terminal.materialization_state
            is SuspendedOperationMaterializationState.CONSUMED
        )
        assert d1.suspended_operation_id == suspended_id
        for gen in (d1, d2, d3):
            _assert_four_ids_match(gen, run_id=run_id, fingerprint=fingerprint)
    finally:
        reset_governed_execution_task(task_token)
        reset_active_execution_governance_identity(gov_token)
        reset_active_execution_identity(id_token)
