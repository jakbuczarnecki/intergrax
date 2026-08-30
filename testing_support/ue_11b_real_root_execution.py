# © Artur Czarnecki. All rights reserved.

"""UE-11B — real root strategy execution proof support (env, stacks, evidence)."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, replace
from typing import Literal

from pydantic import BaseModel, ConfigDict

from intergrax.agents.agent_engine import AgentEngine
from intergrax.agents.authoring.patterns.react import ReActAgent
from intergrax.agents.authoring.patterns.reflex import ReflexAgent
from intergrax.agents.authoring.patterns.types import (
    AgentEvaluation,
    CognitiveEvaluation,
    Observation,
    ReasoningResult,
)
from intergrax.agents.authoring.runtime_tool_helpers import (
    exec_ctx_from_step,
    invoke_catalog_tool,
)
from intergrax.agents.reference_harness import (
    LabHarnessContext,
    build_lab_agent_runtime_config,
    build_lab_agent_runtime_context,
    default_reference_harness,
)
from intergrax.applications._shared.nexus_factory import build_nexus_loop_from_environment
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    OrchestrationProfile,
)
from intergrax.applications.contracts.graph_spec import (
    ApplicationGraphSpec,
    GraphEdge,
    GraphEdgeKind,
    GraphNode,
)
from intergrax.applications.contracts.intent_route import IntentRoute
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.contracts.agent_run_enums import CognitivePattern
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
    require_active_execution_id,
    require_active_execution_identity,
)
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.providers.native_ollama_adapter import NativeOllamaAdapter
from intergrax.runtime.execution import ExecutionCapability, ExecutionRequest, ExecutionResult, ExecutionStatus
from intergrax.runtime.execution.agentic import AgentExecutor
from intergrax.runtime.execution.boundary import ExecutionAdmissionHook
from intergrax.runtime.execution.budget.ledger import (
    InMemoryExecutionBudgetLedger,
    create_execution_budget_ledger,
    fixed_execution_budget_ledger_factory,
)
from intergrax.runtime.execution.budget.models import BudgetUsageTotals
from intergrax.runtime.execution.facade import Execution
from intergrax.runtime.execution.inference import InferenceExecutor
from intergrax.runtime.execution.orchestration import (
    OrchestrationExecutor,
    TaskBoundOrchestrationDelegate,
)
from intergrax.runtime.execution.runtime import ExecutionRuntime, RootExecutionOptions
from intergrax.runtime.execution.strategy_router import StrategyExecutionRouter
from intergrax.runtime.execution.task_adapter import TaskExecutionInput, execution_request_from_task
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.nexus.tracing.persistence_models import PersistedRun, RunMetadata, RunStats
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskResult
from intergrax.tools.providers.harness.bundle import register_harness_tools
from intergrax.tools.providers.harness.service import HARNESS_LIST_RUNS_TOOL_ID
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from testing_support.builder import require_ollama_reachable

UE_11B_PIPELINE_CAPABILITY = "ue11b.orchestration.pipeline"
UE_11B_EXTRACT_CAPABILITY = "ue11b.extract.fact"
UE_11B_CONFIRM_CAPABILITY = "ue11b.confirm.fact"
UE_11B_AGENTIC_CAPABILITY = "ue11b.agentic.tool_loop"
UE_11B_TENANT_ID = "ue11b-proof"
UE_11B_INFERENCE_PROMPT = "Intergrax validates one canonical execution spine."


class TextCategoryClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Literal["platform", "validation", "other"]


class ExecutionIdAdmissionHook:
    __slots__ = ("captured_execution_id", "captured_run_id", "captured_attempt_id")

    def __init__(self) -> None:
        self.captured_execution_id: ExecutionId | None = None
        self.captured_run_id: RunId | None = None
        self.captured_attempt_id: AttemptId | None = None

    async def admit(self, request: object) -> None:
        del request
        run_id, attempt_id = require_active_execution_identity()
        self.captured_run_id = run_id
        self.captured_attempt_id = attempt_id
        self.captured_execution_id = require_active_execution_id()


@dataclass(frozen=True, slots=True)
class Ue11bBudgetEvidence:
    llm_calls: int
    tool_calls: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class Ue11bInferenceStack:
    execution: Execution[
        ExecutionRequest[tuple[ChatMessage, ...], TextCategoryClassification],
        ExecutionResult[TextCategoryClassification],
    ]
    options: RootExecutionOptions
    ledger: InMemoryExecutionBudgetLedger
    adapter: NativeOllamaAdapter
    admission_hook: ExecutionIdAdmissionHook


@dataclass(frozen=True, slots=True)
class Ue11bAgenticStack:
    execution: Execution[
        ExecutionRequest[RuntimeRequest, AgentExecutionResult],
        ExecutionResult[AgentExecutionResult],
    ]
    options: RootExecutionOptions
    ledger: InMemoryExecutionBudgetLedger
    adapter: NativeOllamaAdapter
    agent_id: str
    admission_hook: ExecutionIdAdmissionHook


@dataclass(frozen=True, slots=True)
class Ue11bOrchestrationStack:
    execution: Execution[
        ExecutionRequest[TaskExecutionInput, TaskResult],
        TaskResult,
    ]
    options: RootExecutionOptions
    ledger: InMemoryExecutionBudgetLedger
    adapter: NativeOllamaAdapter
    nexus_loop: NexusLoop
    task: Task
    admission_hook: ExecutionIdAdmissionHook


class _EmptyTraceReader:
    def read_run(self, run_id: str, tenant_id: str) -> PersistedRun:
        del run_id
        return PersistedRun(
            metadata=RunMetadata(
                run_id="ue11b-empty",
                session_id="s-empty",
                user_id="u-empty",
                tenant_id=tenant_id,
                started_at_utc="2026-01-01T00:00:00Z",
                stats=RunStats(duration_ms=0, llm_usage={}),
            ),
            events=[],
        )

    def list_runs(self, tenant_id: str, *, limit: int = 50) -> list[PersistedRun]:
        del tenant_id, limit
        return []


async def _llm_thought_async(
    adapter: LLMAdapter,
    *,
    prompt: str,
    run_id: str,
) -> str:
    def _invoke() -> str:
        response = adapter.generate_messages(
            [ChatMessage(role="user", content=prompt)],
            temperature=0,
            run_id=run_id,
        )
        return response.content.strip()

    return await asyncio.to_thread(_invoke)


def resolve_native_ollama_adapter() -> NativeOllamaAdapter:
    require_ollama_reachable()
    model = os.environ.get("INTERGRAX_LLM_MODEL", NativeOllamaAdapter.DEFAULT_MODEL).strip()
    if not model:
        model = NativeOllamaAdapter.DEFAULT_MODEL
    base_url = os.environ.get("OLLAMA_HOST", "").strip() or None
    return NativeOllamaAdapter(
        model=model,
        base_url=base_url,
        temperature=0,
        num_predict=96,
    )


def _root_options() -> RootExecutionOptions:
    return RootExecutionOptions(
        authority=ParentExecutionAuthority.unrestricted_root(),
        tenant_id=UE_11B_TENANT_ID,
    )


def _harness_tool_surface() -> tuple[ToolRegistry, ToolWiringContext]:
    registry = ToolRegistry()
    wiring = ToolWiringContext(trace_reader=_EmptyTraceReader())
    register_harness_tools(registry, wiring)
    return registry, wiring


def _tool_wiring_context() -> ToolWiringContext:
    _registry, wiring = _harness_tool_surface()
    return wiring


def build_inference_stack() -> Ue11bInferenceStack:
    adapter = resolve_native_ollama_adapter()
    ledger = create_execution_budget_ledger(RunBudget(max_llm_calls=8, max_total_tokens=4000))
    admission_hook = ExecutionIdAdmissionHook()
    router = StrategyExecutionRouter[
        tuple[ChatMessage, ...],
        TextCategoryClassification,
        ExecutionResult[TextCategoryClassification],
    ](inference_executor=InferenceExecutor(adapter))
    runtime = ExecutionRuntime[
        ExecutionRequest[tuple[ChatMessage, ...], TextCategoryClassification],
        ExecutionResult[TextCategoryClassification],
    ](
        router,
        ledger_factory=fixed_execution_budget_ledger_factory(ledger),
        run_budget=RunBudget(max_llm_calls=8, max_total_tokens=4000),
        admission_hooks=(admission_hook,),
    )
    return Ue11bInferenceStack(
        execution=Execution(runtime),
        options=_root_options(),
        ledger=ledger,
        adapter=adapter,
        admission_hook=admission_hook,
    )


class Ue11bToolReActAgent(ReActAgent):
    contract_id = "ue11b_tool_react"
    capabilities = (UE_11B_AGENTIC_CAPABILITY,)
    agent_name = "UE-11B Tool ReAct Proof"
    agent_description = "Real LLM + read-only harness.list_runs tool proof agent."
    agent_version = "1.0.0"
    risk_level = AgentRiskLevel.LOW
    default_max_react_iterations = 3

    def __init__(self, adapter: LLMAdapter, harness: LabHarnessContext) -> None:
        self._adapter = adapter
        self._harness = harness
        self._tool_invoked = False

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=self.contract_id,
            name=self.agent_name,
            description=self.agent_description,
            version=self.agent_version,
            capabilities=list(self.capabilities),
            allowed_tools=[HARNESS_LIST_RUNS_TOOL_ID],
            risk_level=self.risk_level,
            lifecycle_state=AgentLifecycleState.PRODUCTION,
            production_eligible=True,
            owner_team="platform",
            owner_contact="harness@intergrax",
            on_call_contact="harness@intergrax",
            runbook_ref="docs/project/architecture/intergrax_runtime_architecture.md",
            cognitive_pattern=self.cognitive_pattern,
            pattern_version=self.pattern_version,
            max_steps=self.default_max_react_iterations,
        )

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        capability = task_context.capability
        if capability in (None, UE_11B_AGENTIC_CAPABILITY):
            return CapabilityMatchResult(
                matched=True,
                agent_id=self.contract_id,
                matched_capabilities=[UE_11B_AGENTIC_CAPABILITY],
                score=1.0,
                rationale="ue11b agentic proof",
            )
        return CapabilityMatchResult(matched=False, rationale="capability not supported")

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        registry, wiring = _harness_tool_surface()
        harness = replace(self._harness, tool_wiring_context=wiring)
        config = build_lab_agent_runtime_config(
            request=request,
            llm_adapter=self._adapter,
            harness=harness,
        )
        config.tool_registry = registry
        return RuntimeContext.build(
            config=config,
            session_manager=SessionManager(storage=InMemorySessionStorage()),
        )

    async def perceive(self, step_ctx: AgentStepContext) -> Observation:
        return Observation(summary=step_ctx.message or UE_11B_INFERENCE_PROMPT)

    async def reason(
        self,
        step_ctx: AgentStepContext,
        observation: Observation,
    ) -> ReasoningResult:
        prompt = (
            "Reply with one short sentence confirming you received this platform proof input: "
            f"{observation.summary}"
        )
        thought = await _llm_thought_async(
            self._adapter,
            prompt=prompt,
            run_id=step_ctx.run_id,
        )
        return ReasoningResult(thought=thought)

    async def act(
        self,
        step_ctx: AgentStepContext,
        reasoning: ReasoningResult,
    ) -> dict[str, object]:
        exec_ctx = exec_ctx_from_step(step_ctx)
        if exec_ctx is None:
            raise RuntimeError("UAEP exec context missing for UE-11B agentic proof")
        tool_result = await invoke_catalog_tool(
            exec_ctx,
            tool_name=HARNESS_LIST_RUNS_TOOL_ID,
            agent_id=self.contract_id,
            step_id=self.main_step_id,
            tool_input={"tenant_id": UE_11B_TENANT_ID, "limit": 1},
        )
        self._tool_invoked = True
        tool_status = str(tool_result.get("status", ""))
        return {
            "summary": f"{reasoning.thought} [tool:{tool_status}]",
            "tool_status": tool_status,
            "tool_calls": 1,
        }

    def evaluate(
        self,
        step_ctx: AgentStepContext,
        output: dict[str, object],
    ) -> AgentEvaluation:
        del step_ctx
        if self._tool_invoked and output.get("summary"):
            return AgentEvaluation(
                verdict=CognitiveEvaluation.COMPLETE,
                reason="tool_feedback_received",
            )
        return AgentEvaluation(verdict=CognitiveEvaluation.CONTINUE, reason="await_tool")


def build_agentic_stack() -> Ue11bAgenticStack:
    adapter = resolve_native_ollama_adapter()
    harness = replace(
        default_reference_harness(),
        tool_wiring_context=_tool_wiring_context(),
    )
    agent = Ue11bToolReActAgent(adapter, harness)
    registry = AgentRegistry()
    registry.register(agent)
    engine = AgentEngine(registry)
    ledger = create_execution_budget_ledger(RunBudget(max_llm_calls=8, max_tool_calls=4, max_total_tokens=4000))
    admission_hook = ExecutionIdAdmissionHook()
    router = StrategyExecutionRouter[
        RuntimeRequest,
        AgentExecutionResult,
        ExecutionResult[AgentExecutionResult],
    ](agent_executor=AgentExecutor(engine))
    runtime = ExecutionRuntime[
        ExecutionRequest[RuntimeRequest, AgentExecutionResult],
        ExecutionResult[AgentExecutionResult],
    ](
        router,
        ledger_factory=fixed_execution_budget_ledger_factory(ledger),
        run_budget=RunBudget(max_llm_calls=8, max_tool_calls=4, max_total_tokens=4000),
        admission_hooks=(admission_hook,),
    )
    return Ue11bAgenticStack(
        execution=Execution(runtime),
        options=_root_options(),
        ledger=ledger,
        adapter=adapter,
        agent_id=agent.contract_id,
        admission_hook=admission_hook,
    )


def agentic_runtime_request(*, run_id: RunId) -> RuntimeRequest:
    return RuntimeRequest(
        tenant_id=UE_11B_TENANT_ID,
        user_id="ue11b-user",
        session_id="ue11b-session",
        agent_id=Ue11bToolReActAgent.contract_id,
        message=(
            "Use the harness list-runs tool once, then answer with a short confirmation "
            "that the platform proof completed."
        ),
        task_id=mint_task_id(),
        run_id=run_id,
        metadata={},
    )


def correlated_agentic_inputs() -> tuple[RunId, RootExecutionOptions, RuntimeRequest]:
    run_id = mint_run_id()
    options = RootExecutionOptions(
        authority=ParentExecutionAuthority.unrestricted_root(),
        tenant_id=UE_11B_TENANT_ID,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
    )
    return run_id, options, agentic_runtime_request(run_id=run_id)


class _Ue11bOrchestrationReflexAgent(ReflexAgent):
    cognitive_pattern = CognitivePattern.REFLEX

    def __init__(
        self,
        *,
        contract_id: str,
        capability: str,
        adapter: LLMAdapter,
        harness: LabHarnessContext,
        prompt_template: str,
    ) -> None:
        self.contract_id = contract_id
        self.capabilities = (capability,)
        self.agent_name = contract_id
        self.agent_description = "UE-11B orchestration proof agent"
        self.agent_version = "1.0.0"
        self.risk_level = AgentRiskLevel.LOW
        self._adapter = adapter
        self._harness = harness
        self._prompt_template = prompt_template

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=self.contract_id,
            name=self.agent_name,
            description=self.agent_description,
            version=self.agent_version,
            capabilities=list(self.capabilities),
            risk_level=self.risk_level,
            lifecycle_state=AgentLifecycleState.PRODUCTION,
            production_eligible=True,
            owner_team="platform",
            owner_contact="harness@intergrax",
            on_call_contact="harness@intergrax",
            runbook_ref="docs/project/architecture/intergrax_runtime_architecture.md",
            cognitive_pattern=self.cognitive_pattern,
            pattern_version=self.pattern_version,
            max_steps=3,
        )

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        capability = task_context.capability
        supported = set(self.capabilities) | {UE_11B_PIPELINE_CAPABILITY}
        if capability is None or capability in supported:
            return CapabilityMatchResult(
                matched=True,
                agent_id=self.contract_id,
                matched_capabilities=list(supported),
                score=1.0,
                rationale="ue11b orchestration proof",
            )
        return CapabilityMatchResult(matched=False, rationale="capability not supported")

    def build_context(self, request: RuntimeRequest) -> object:
        return build_lab_agent_runtime_context(
            request=request,
            llm_adapter=self._adapter,
            harness=self._harness,
        )

    async def perceive(self, step_ctx: AgentStepContext) -> Observation:
        return Observation(summary=step_ctx.message or "")

    async def reason(
        self,
        step_ctx: AgentStepContext,
        observation: Observation,
    ) -> ReasoningResult:
        prompt = self._prompt_template.format(message=observation.summary)
        thought = await _llm_thought_async(
            self._adapter,
            prompt=prompt,
            run_id=step_ctx.run_id,
        )
        return ReasoningResult(thought=thought)

    async def act(
        self,
        step_ctx: AgentStepContext,
        reasoning: ReasoningResult,
    ) -> dict[str, object]:
        del step_ctx
        answer = reasoning.thought
        return {"summary": answer, "answer": answer}

    def evaluate(
        self,
        step_ctx: AgentStepContext,
        output: dict[str, object],
    ) -> AgentEvaluation:
        del step_ctx, output
        return AgentEvaluation(verdict=CognitiveEvaluation.COMPLETE, reason="ue11b_orchestration_step")


def _orchestration_environment() -> ApplicationEnvironmentProfile:
    return ApplicationEnvironmentProfile.lab_defaults(profile_id="ue11b.orchestration").model_copy(
        update={
            "graph_spec": ApplicationGraphSpec(
                nodes=[
                    GraphNode(agent_id="ue11b_extract"),
                    GraphNode(agent_id="ue11b_confirm"),
                ],
                edges=[
                    GraphEdge(
                        source_agent_id="ue11b_extract",
                        target_agent_id="ue11b_confirm",
                        kind=GraphEdgeKind.DEPENDS_ON,
                    ),
                ],
                trigger_capabilities=[UE_11B_PIPELINE_CAPABILITY],
            ),
            "orchestration_profile": OrchestrationProfile(
                classifier_kind="rules",
                merge_strategy="structured_json",
                intent_routes=[
                    IntentRoute(
                        capability=UE_11B_PIPELINE_CAPABILITY,
                        keywords=["canonical", "execution", "spine", "intergrax"],
                    ),
                ],
            ),
        }
    )


def build_orchestration_stack() -> Ue11bOrchestrationStack:
    adapter = resolve_native_ollama_adapter()
    harness = default_reference_harness()
    registry = AgentRegistry()
    registry.register(
        _Ue11bOrchestrationReflexAgent(
            contract_id="ue11b_extract",
            capability=UE_11B_EXTRACT_CAPABILITY,
            adapter=adapter,
            harness=harness,
            prompt_template=(
                "Extract one keyword from this message about platform execution. "
                "Reply with only the single word 'canonical': {message}"
            ),
        )
    )
    registry.register(
        _Ue11bOrchestrationReflexAgent(
            contract_id="ue11b_confirm",
            capability=UE_11B_CONFIRM_CAPABILITY,
            adapter=adapter,
            harness=harness,
            prompt_template=(
                "If the prior agent output contains the word canonical, reply YES-canonical. "
                "Input context: {message}"
            ),
        )
    )
    ledger = create_execution_budget_ledger(RunBudget(max_llm_calls=12, max_total_tokens=6000))
    nexus_loop = build_nexus_loop_from_environment(
        registry,
        env=_orchestration_environment(),
        llm_adapter=adapter,
        execution_budget_ledger=ledger,
        run_budget=RunBudget(max_llm_calls=12, max_total_tokens=6000),
    )
    task = Task(
        tenant_id=UE_11B_TENANT_ID,
        user_id="ue11b-user",
        message=UE_11B_INFERENCE_PROMPT,
        context=TaskContext(capability=UE_11B_PIPELINE_CAPABILITY),
        metadata={},
    )
    delegate = TaskBoundOrchestrationDelegate(
        task,
        OrchestrationExecutor(nexus_loop),
    )
    admission_hook = ExecutionIdAdmissionHook()
    router = StrategyExecutionRouter[
        TaskExecutionInput,
        TaskResult,
        TaskResult,
    ](orchestration_executor=delegate)
    runtime = ExecutionRuntime[
        ExecutionRequest[TaskExecutionInput, TaskResult],
        TaskResult,
    ](
        router,
        ledger_factory=fixed_execution_budget_ledger_factory(ledger),
        run_budget=RunBudget(max_llm_calls=12, max_total_tokens=6000),
        admission_hooks=(admission_hook,),
    )
    return Ue11bOrchestrationStack(
        execution=Execution(runtime),
        options=_root_options(),
        ledger=ledger,
        adapter=adapter,
        nexus_loop=nexus_loop,
        task=task,
        admission_hook=admission_hook,
    )


def orchestration_request(task: Task) -> ExecutionRequest[TaskExecutionInput, TaskResult]:
    return execution_request_from_task(
        task,
        capabilities=frozenset({ExecutionCapability.ORCHESTRATION}),
        output_type=TaskResult,
    )


def budget_evidence(
    ledger: InMemoryExecutionBudgetLedger,
    *,
    attempt_id: AttemptId,
) -> Ue11bBudgetEvidence:
    snapshot = ledger.export_snapshot(attempt_id)
    consumed = snapshot.root_shared_consumed.add(snapshot.root_permanent_consumed)
    for record in snapshot.records:
        consumed = consumed.add(record.consumed)
    return Ue11bBudgetEvidence(
        llm_calls=consumed.llm_calls,
        tool_calls=consumed.tool_calls,
        total_tokens=consumed.total_tokens,
    )


def child_execution_records(
    ledger: InMemoryExecutionBudgetLedger,
    *,
    attempt_id: AttemptId,
    root_execution_id: ExecutionId,
) -> tuple[ExecutionId, ...]:
    snapshot = ledger.export_snapshot(attempt_id)
    child_ids: list[ExecutionId] = []
    for record in snapshot.records:
        if record.parent_execution_id == root_execution_id and record.execution_id != root_execution_id:
            child_ids.append(record.execution_id)
    return tuple(child_ids)


def assert_completed_inference_result(
    result: ExecutionResult[TextCategoryClassification],
) -> TextCategoryClassification:
    assert result.status is ExecutionStatus.COMPLETED
    assert result.output is not None
    assert result.output.category in {"platform", "validation", "other"}
    return result.output


def assert_completed_agentic_result(
    result: ExecutionResult[AgentExecutionResult],
) -> AgentExecutionResult:
    assert result.status is ExecutionStatus.COMPLETED
    assert result.output.status.value == "completed"
    assert result.output.summary
    return result.output
