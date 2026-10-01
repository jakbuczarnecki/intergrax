# © Artur Czarnecki. All rights reserved.

"""GR-13 per-GEP GR-8 fact emission on canonical production owners."""

from __future__ import annotations

from contextvars import Token
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

if TYPE_CHECKING:
    from intergrax.runtime.governance.governance_evidence_recorder import (
        GovernanceEvidenceRecorder,
    )
    from intergrax.runtime.governance.governance_evidence_persistence import (
        InMemoryGovernanceEvidencePersistence,
    )

from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications.contracts.environment_profile.root import (
    ApplicationEnvironmentProfile,
)
from intergrax.applications.contracts.environment_profile.sub_profiles import (
    PolicyRulesProfile,
)
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.execution_identity import (
    ActiveExecutionIdentity,
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.contracts.governed_execution_governance_evidence import (
    GovernedExecutionEvaluationPoint,
    GovernanceDecisionEvidenceFact,
)
from intergrax.contracts.policy_enforcement_mode import PolicyEnforcementMode
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.contracts.runtime_policy_context import (
    AgentDecisionPolicyContext,
    PreModelPolicyContext,
)
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.governance.execution_guard import GovernanceEvaluation
from intergrax.runtime.governance.governance_evidence_composition import (
    build_governance_evidence_recorder,
    build_in_memory_governance_evidence_persistence,
)
from intergrax.runtime.governance.post_run_governance_bridge import (
    invoke_post_run_governance,
)
from intergrax.runtime.interrupts.handler import ExecutionInterruptHandler
from intergrax.runtime.kernel.step_kernel import HarnessKernel, StepKernelContext
from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.orchestration.hitl_runner import NexusHitlRunner
from intergrax.runtime.nexus.orchestration.planning_runner import NexusPlanningRunner
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.chat_session import ChatSession
from intergrax.runtime.nexus.task_classifier import TaskClassification
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.nexus.tools.tool_runtime import ToolInvocationPlan, ToolRuntime
from intergrax.runtime.policy.policy_engine import PolicyEngine
from intergrax.runtime.policy.pre_model_policy_evaluation import (
    enforce_pre_model_before_structured_inference,
)
from intergrax.runtime.policy.pre_output_policy_bridge import apply_pre_output_policy
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine
from intergrax.runtime.replay.metrics import ExecutionMetrics
from intergrax.runtime.replay.policy import PolicyDecision as ReplayPolicyDecision
from intergrax.runtime.replay.policy import PolicyDecisionType
from intergrax.runtime.replay.regression import RegressionSignals
from intergrax.runtime.task.task import Task, TaskResult, TaskState
from intergrax.runtime.task.task_lifecycle import TaskLifecycle
from intergrax.runtime.task.task_trace import TaskTraceEmitter
from intergrax.runtime.tools.scope_policy import ToolScopePolicy
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.unified.constants import (
    RAG_RETRIEVE_TOOL_ID,
    WEBSEARCH_QUERY_TOOL_ID,
)
from intergrax.runtime.nexus.errors.declarative_policy_violation_error import (
    DeclarativePolicyViolationError,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from testing_support.builder import (
    FakeLLMAdapter,
    build_in_memory_session_manager,
    canonical_execution_identity_scope,
    canonical_run_id_for_tests,
    kernel_step_test_scope,
    tools_agent_make_contract,
)

_TENANT = "tenant-gr13"
_WORKSPACE = "workspace-gr13"
_PRINCIPAL = "principal-gr13"
_AGENT = "agent_1"
_PROBE_TOOL = "probe.tool"


class _DenyPreLlm(RuntimePolicyEngine):
    def evaluate_pre_llm(
        self,
        *,
        tenant_id: str,
        principal_id: str,
        agent_id: str | None = None,
        message_count: int = 0,
        context: PreModelPolicyContext | None = None,
    ) -> PolicyDecision:
        return PolicyDecision(
            action=PolicyAction.DENY,
            reason="pre_model_denied",
            policy_rule_id="test.gr13.pre_model_deny",
        )


class _DenyContinue(RuntimePolicyEngine):
    def evaluate_decision(
        self,
        decision: AgentDecision,
        *,
        context: AgentDecisionPolicyContext | None = None,
    ) -> PolicyDecision:
        return PolicyDecision(
            action=PolicyAction.DENY,
            reason="denied",
            policy_rule_id="test.deny",
        )


class _RequireHumanContinue(RuntimePolicyEngine):
    def evaluate_decision(
        self,
        decision: AgentDecision,
        *,
        context: AgentDecisionPolicyContext | None = None,
    ) -> PolicyDecision:
        return PolicyDecision(
            action=PolicyAction.REQUIRE_HUMAN,
            reason="human_required",
            policy_rule_id="test.require_human",
        )


class _DenyAllToolsPolicy(ToolScopePolicy):
    def is_allowed(self, *, agent_id: str, tool_id: str) -> bool:
        return False


class _In(BaseModel):
    query: str = "q"


class _Out(BaseModel):
    used: bool = True
    context_text: str = "ctx"


class _OkHandler:
    def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
        return _Out()


class _ClassifyOk:
    def classify(self, task: Task) -> Task:
        task.runtime.classification.value = (
            TaskClassification.SINGLE_AGENT_DEFAULT.value
        )
        return task


def assert_gr13_fact_identity(
    fact: GovernanceDecisionEvidenceFact,
    *,
    evaluation_point: GovernedExecutionEvaluationPoint,
    decision: PolicyAction,
    tenant_id: str = _TENANT,
    workspace_id: str = _WORKSPACE,
    principal_id: str = _PRINCIPAL,
) -> None:
    assert fact.evaluation_point is evaluation_point
    assert fact.decision is decision
    assert fact.tenant_id == tenant_id
    assert fact.workspace_id == workspace_id
    assert fact.principal_id == principal_id


def _governance_context() -> tuple[Token, Token]:
    gov = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            principal_id=_PRINCIPAL,
        ),
    )
    run = mint_run_id()
    attempt = mint_attempt_id()
    execution = mint_execution_id()
    ident = bind_active_execution_identity(
        run_id=run,
        attempt_id=attempt,
        execution_id=execution,
    )
    return gov, ident


@pytest.fixture
def gr13_evidence_store():
    store = build_in_memory_governance_evidence_persistence()
    recorder = build_governance_evidence_recorder(persistence=store)
    gov, ident = _governance_context()
    yield store, recorder
    reset_active_execution_governance_identity(gov)
    reset_active_execution_identity(ident)


def _tool_runtime_state(
    recorder: GovernanceEvidenceRecorder,
    *,
    tool_scope_policy: ToolScopePolicy | None = None,
) -> RuntimeState:
    run_id = canonical_run_id_for_tests("gr13-tool-plan")
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        production_mode=False,
        enable_rag=True,
        tool_scope_policy=tool_scope_policy,
    )
    ctx = RuntimeContext(
        config=config,
        session_manager=build_in_memory_session_manager(),
        prompt_registry=MagicMock(),
        context_builder=MagicMock(),
        rag_prompt_builder=MagicMock(),
        websearch_executor=MagicMock(),
        websearch_prompt_builder=MagicMock(),
    )
    ctx.governance_evidence_recorder = cast(GovernanceEvidenceRecorder, recorder)
    return RuntimeState(
        context=ctx,
        request=RuntimeRequest(
            agent_id=_AGENT,
            user_id=_PRINCIPAL,
            session_id="session-gr13",
            tenant_id=_TENANT,
            message="plan tools",
            task_id=mint_task_id(),
            run_id=run_id,
        ),
        run_id=run_id,
        session=ChatSession(
            id="session-gr13",
            tenant_id=_TENANT,
            user_id=_PRINCIPAL,
        ),
        tool_traces=[],
    )


async def _prove_tool_plan_or_access_emits_fact(
    store: InMemoryGovernanceEvidencePersistence,
    recorder: GovernanceEvidenceRecorder,
) -> None:
    state = _tool_runtime_state(recorder, tool_scope_policy=_DenyAllToolsPolicy())
    plan = ToolInvocationPlan.from_tool_ids(
        [RAG_RETRIEVE_TOOL_ID, WEBSEARCH_QUERY_TOOL_ID],
    )
    with canonical_execution_identity_scope(state.run_id):
        await ToolRuntime.invoke(state=state, plan=plan, allowed_tools=())
    assert len(store.facts) == 1
    fact = store.facts[0]
    assert fact.evaluation_point is GovernedExecutionEvaluationPoint.TOOL_PLAN_OR_ACCESS
    assert fact.decision is PolicyAction.DENY
    assert fact.tenant_id == _TENANT
    assert fact.workspace_id == _TENANT
    assert fact.principal_id == _PRINCIPAL


def _policy_bundle_deny_tool(tool_id: str) -> RuntimePolicyBundle:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="gr13.tool.policy")
    env = env.model_copy(
        update={
            "policy_rules": PolicyRulesProfile(
                inline_rules=[
                    {
                        "rule_id": "gr13.deny.tool",
                        "handler_id": "deny_tool",
                        "resource_kind": "tool",
                        "resource_id": tool_id,
                        "action": "deny",
                    }
                ],
                policy_enforcement_mode=PolicyEnforcementMode.ENFORCE,
            ),
        },
    )
    return wire_policy_bundle(env)


def _prove_tool_invocation_policy_emits_fact(
    store: InMemoryGovernanceEvidencePersistence,
    recorder: GovernanceEvidenceRecorder,
) -> None:
    registry = ToolRegistry()
    contract = tools_agent_make_contract(_PROBE_TOOL, _In, _Out)
    registry.register(contract, _OkHandler())
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=RegistryToolExecutor(registry),
        governance_evidence_recorder=recorder,
    )
    state = _tool_runtime_state(recorder)
    state.context.config.policy_bundle = _policy_bundle_deny_tool(_PROBE_TOOL)
    run_id = state.run_id
    request = ToolExecutionRequest(
        run_id=run_id,
        step_id="step-gr13",
        tool_id=_PROBE_TOOL,
        input=_In(),
        idempotency_key="gr13-tool-invoke",
    )
    with canonical_execution_identity_scope(run_id):
        with pytest.raises(DeclarativePolicyViolationError):
            invoker.invoke(
                state=state,
                agent_id=_AGENT,
                request=cast(ToolExecutionRequest[BaseModel], request),
            )
    assert len(store.facts) == 1
    fact = store.facts[0]
    assert_gr13_fact_identity(
        fact,
        evaluation_point=GovernedExecutionEvaluationPoint.TOOL_INVOCATION_POLICY,
        decision=PolicyAction.DENY,
    )
    assert fact.run_id is not None


@pytest.mark.unit
def test_gr13_agentic_pre_model_emits_fact(gr13_evidence_store) -> None:
    store, recorder = gr13_evidence_store
    enforce_pre_model_before_structured_inference(
        PolicyEngine(),
        recorder,
        adapter=FakeLLMAdapter(),
        messages=[ChatMessage(role="user", content="hi")],
        inference_profile_id=None,
    )
    assert len(store.facts) == 1
    assert_gr13_fact_identity(
        store.facts[0],
        evaluation_point=GovernedExecutionEvaluationPoint.PRE_MODEL,
        decision=PolicyAction.ALLOW,
    )


@pytest.mark.unit
def test_gr13_agentic_agent_decision_emits_fact(gr13_evidence_store) -> None:
    store, recorder = gr13_evidence_store
    task_id = str(mint_task_id())
    run_id = str(mint_run_id())
    handler = ExecutionInterruptHandler(
        PolicyEngine(),
        governance_evidence_recorder=recorder,
    )
    resolution = handler.resolve_decision(
        AgentDecision(type=AgentDecisionType.CONTINUE, reason="ok"),
        task_id=task_id,
        run_id=run_id,
        agent_id=_AGENT,
    )
    assert resolution.policy_decision.action is PolicyAction.ALLOW
    fact = next(
        f
        for f in store.facts
        if f.evaluation_point is GovernedExecutionEvaluationPoint.AGENT_DECISION
    )
    assert_gr13_fact_identity(
        fact,
        evaluation_point=GovernedExecutionEvaluationPoint.AGENT_DECISION,
        decision=PolicyAction.ALLOW,
    )
    assert fact.task_id is not None
    assert str(fact.task_id) == task_id
    assert str(fact.run_id) == run_id


@pytest.mark.unit
def test_gr13_agentic_interrupt_emits_fact(gr13_evidence_store) -> None:
    store, recorder = gr13_evidence_store
    handler = ExecutionInterruptHandler(
        PolicyEngine(),
        governance_evidence_recorder=recorder,
    )
    handler.resolve_decision(
        AgentDecision(type=AgentDecisionType.INTERRUPT, reason="stop", payload={}),
        task_id=str(mint_task_id()),
        run_id=str(mint_run_id()),
        agent_id=_AGENT,
    )
    fact = next(
        f
        for f in store.facts
        if f.evaluation_point is GovernedExecutionEvaluationPoint.INTERRUPT
    )
    assert fact.evaluation_point is GovernedExecutionEvaluationPoint.INTERRUPT
    assert fact.decision in {
        PolicyAction.ALLOW,
        PolicyAction.REQUIRE_HUMAN,
        PolicyAction.DENY,
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_gr13_agentic_tool_plan_or_access_emits_fact(gr13_evidence_store) -> None:
    store, recorder = gr13_evidence_store
    await _prove_tool_plan_or_access_emits_fact(store, recorder)


@pytest.mark.unit
def test_gr13_agentic_tool_invocation_policy_emits_fact(gr13_evidence_store) -> None:
    store, recorder = gr13_evidence_store
    _prove_tool_invocation_policy_emits_fact(store, recorder)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_gr13_agentic_pre_output_emits_fact(gr13_evidence_store) -> None:
    store, recorder = gr13_evidence_store
    with kernel_step_test_scope("gr13-pre-output") as (task_id, run_id):
        kernel_ctx = StepKernelContext(
            agent_id=_AGENT,
            task_id=task_id,
            run_id=run_id,
            tenant_id=_TENANT,
            principal_id=_PRINCIPAL,
            policy_engine=PolicyEngine(),
            governance_evidence_recorder=recorder,
            allow_permissive_missing_policy=True,
        )
        step_ctx = AgentStepContext(tenant_id="tenant-test", step_index=0)
        outcome = StepOutcome.complete("done")
        await HarnessKernel.execute_step(outcome, step_ctx, kernel_ctx)
    assert len(store.facts) == 1
    fact = store.facts[0]
    assert fact.evaluation_point is GovernedExecutionEvaluationPoint.PRE_OUTPUT
    assert fact.decision is PolicyAction.ALLOW
    assert fact.tenant_id == _TENANT
    assert fact.principal_id == _PRINCIPAL
    assert str(fact.run_id) == run_id


@pytest.mark.unit
def test_gr13_agentic_post_run_emits_fact(gr13_evidence_store) -> None:
    store, recorder = gr13_evidence_store
    run_id = str(mint_run_id())

    class _PostRun:
        def evaluate(self, run_id: str, agent_id: str) -> GovernanceEvaluation:
            return GovernanceEvaluation(
                decision=ReplayPolicyDecision(
                    decision=PolicyDecisionType.ALLOW,
                    reasons=[],
                ),
                metrics=ExecutionMetrics(
                    step_count=1,
                    total_llm_calls=0,
                    total_tool_calls=0,
                    total_artifacts=0,
                    total_tokens=0,
                    duration=None,
                    tool_steps_ratio=0.0,
                    llm_steps_ratio=0.0,
                ),
                regression=RegressionSignals(),
            )

    invoke_post_run_governance(
        _PostRun(),
        run_id=run_id,
        agent_id=_AGENT,
        governance_evidence_recorder=recorder,
    )
    from intergrax.contracts.execution_identity import require_active_execution_identity

    assert_gr13_fact_identity(
        store.facts[0],
        evaluation_point=GovernedExecutionEvaluationPoint.POST_RUN,
        decision=PolicyAction.ALLOW,
    )
    active_run, _ = require_active_execution_identity()
    assert store.facts[0].run_id == active_run


@pytest.mark.unit
@pytest.mark.asyncio
async def test_gr13_orchestration_pre_model_emits_fact(gr13_evidence_store) -> None:
    store, recorder = gr13_evidence_store
    task = Task(
        tenant_id=_TENANT,
        user_id=_PRINCIPAL,
        message="plan",
        agent_id=_AGENT,
    )
    task.metadata["workspace_id"] = _WORKSPACE
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    ident = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=mint_execution_id(),
        task_id=task.task_id,
    )
    event_bus = RuntimeEventBus()
    metric_scope = event_bus.open_runtime_event_metric_scope(task.task_id, run_id)

    async def _finish(
        task: Task,
        trace_emitter: TaskTraceEmitter,
        *,
        answer: str,
        executions: list,
        validation,
        plan,
        retry_records: list,
        graph_id: str,
        runtime_event_metric_scope,
    ) -> TaskResult:
        return TaskResult(
            task_id=str(task.task_id),
            state=TaskState.FAILED,
            answer=answer,
        )

    noop_hitl = cast(NexusHitlRunner, MagicMock())
    noop_hitl.run_lifecycle_hook = AsyncMock(return_value=None)

    runner = NexusPlanningRunner(
        classifier=_ClassifyOk(),
        planner=MagicMock(),
        registry=MagicMock(),
        hitl=noop_hitl,
        publish=AsyncMock(),
        finish_task=_finish,
        maybe_checkpoint=AsyncMock(),
        policy_engine=PolicyEngine(runtime=_DenyPreLlm()),
        execution_identity=ActiveExecutionIdentity(),
        governance_evidence_recorder=recorder,
    )
    trace_emitter = TaskTraceEmitter(run_id=run_id, attempt_id=attempt_id)
    await runner.run(
        task,
        lifecycle=TaskLifecycle(),
        trace_emitter=trace_emitter,
        runtime_event_metric_scope=metric_scope,
    )
    reset_active_execution_identity(ident)
    assert len(store.facts) == 1
    assert_gr13_fact_identity(
        store.facts[0],
        evaluation_point=GovernedExecutionEvaluationPoint.PRE_MODEL,
        decision=PolicyAction.DENY,
    )
    assert str(store.facts[0].task_id) == str(task.task_id)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_gr13_orchestration_tool_plan_or_access_emits_fact(
    gr13_evidence_store,
) -> None:
    store, recorder = gr13_evidence_store
    await _prove_tool_plan_or_access_emits_fact(store, recorder)


@pytest.mark.unit
def test_gr13_orchestration_tool_invocation_policy_emits_fact(
    gr13_evidence_store,
) -> None:
    store, recorder = gr13_evidence_store
    _prove_tool_invocation_policy_emits_fact(store, recorder)


@pytest.mark.unit
def test_gr13_orchestration_pre_output_emits_fact(gr13_evidence_store) -> None:
    store, recorder = gr13_evidence_store
    task = Task(tenant_id=_TENANT, user_id=_PRINCIPAL, message="hi", agent_id=_AGENT)
    task.metadata["workspace_id"] = _WORKSPACE
    bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        task_id=task.task_id,
    )
    apply_pre_output_policy(
        PolicyEngine(),
        task,
        answer="done",
        governance_evidence_recorder=recorder,
    )
    assert_gr13_fact_identity(
        store.facts[0],
        evaluation_point=GovernedExecutionEvaluationPoint.PRE_OUTPUT,
        decision=PolicyAction.ALLOW,
    )
    assert str(store.facts[0].task_id) == str(task.task_id)


@pytest.mark.unit
def test_gr13_orchestration_post_run_emits_fact(gr13_evidence_store) -> None:
    store, recorder = gr13_evidence_store
    run_id = str(mint_run_id())

    class _PostRun:
        def evaluate(self, run_id: str, agent_id: str) -> GovernanceEvaluation:
            return GovernanceEvaluation(
                decision=ReplayPolicyDecision(
                    decision=PolicyDecisionType.BLOCK,
                    reasons=["post_run_block"],
                ),
                metrics=ExecutionMetrics(
                    step_count=1,
                    total_llm_calls=0,
                    total_tool_calls=0,
                    total_artifacts=0,
                    total_tokens=0,
                    duration=None,
                    tool_steps_ratio=0.0,
                    llm_steps_ratio=0.0,
                ),
                regression=RegressionSignals(),
            )

    invoke_post_run_governance(
        _PostRun(),
        run_id=run_id,
        agent_id=_AGENT,
        governance_evidence_recorder=recorder,
    )
    assert_gr13_fact_identity(
        store.facts[0],
        evaluation_point=GovernedExecutionEvaluationPoint.POST_RUN,
        decision=PolicyAction.DENY,
    )


@pytest.mark.unit
def test_gr13_evidence_does_not_grant_permission(gr13_evidence_store) -> None:
    _, recorder = gr13_evidence_store
    handler = ExecutionInterruptHandler(
        _DenyContinue(),
        governance_evidence_recorder=recorder,
    )
    resolution = handler.resolve_decision(
        AgentDecision(type=AgentDecisionType.CONTINUE, reason="ok"),
        task_id=str(mint_task_id()),
        run_id=str(mint_run_id()),
        agent_id=_AGENT,
    )
    assert resolution.should_block_execution is True
    assert resolution.policy_decision.action is PolicyAction.DENY


@pytest.mark.unit
def test_gr13_require_human_still_blocks_after_evidence(gr13_evidence_store) -> None:
    _, recorder = gr13_evidence_store
    handler = ExecutionInterruptHandler(
        _RequireHumanContinue(),
        governance_evidence_recorder=recorder,
    )
    resolution = handler.resolve_decision(
        AgentDecision(type=AgentDecisionType.CONTINUE, reason="ok"),
        task_id=str(mint_task_id()),
        run_id=str(mint_run_id()),
        agent_id=_AGENT,
    )
    assert resolution.policy_decision.action is PolicyAction.REQUIRE_HUMAN
    assert resolution.should_pause is True


@pytest.mark.unit
def test_gr13_recorder_absence_does_not_grant_permission() -> None:
    handler = ExecutionInterruptHandler(
        _DenyContinue(), governance_evidence_recorder=None
    )
    resolution = handler.resolve_decision(
        AgentDecision(type=AgentDecisionType.CONTINUE, reason="ok"),
        task_id=str(mint_task_id()),
        run_id=str(mint_run_id()),
        agent_id=_AGENT,
    )
    assert resolution.should_block_execution is True


@pytest.mark.unit
def test_gr13_evidence_persistence_failure_does_not_widen_authority(
    gr13_evidence_store,
) -> None:
    store, _ = gr13_evidence_store
    store.fail_on_persist = True
    recorder = build_governance_evidence_recorder(persistence=store)
    handler = ExecutionInterruptHandler(
        _DenyContinue(),
        governance_evidence_recorder=recorder,
    )
    resolution = handler.resolve_decision(
        AgentDecision(type=AgentDecisionType.CONTINUE, reason="ok"),
        task_id=str(mint_task_id()),
        run_id=str(mint_run_id()),
        agent_id=_AGENT,
    )
    assert resolution.should_block_execution is True
    assert resolution.policy_decision.action is PolicyAction.DENY
