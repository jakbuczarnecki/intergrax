# © Artur Czarnecki. All rights reserved.

"""DG-001 canonical root execution + terminal diagnostics qualification harness."""

from __future__ import annotations

from dataclasses import dataclass, replace

from intergrax.agents.authoring.patterns.reflex import ReflexAgent
from intergrax.agents.reference_harness import (
    build_lab_agent_runtime_context,
    default_reference_harness,
)
from intergrax.agents.authoring.patterns.types import (
    AgentEvaluation,
    CognitiveEvaluation,
    Observation,
    ReasoningResult,
)
from intergrax.applications._shared.diagnostic_read_wiring import (
    HostDiagnosticReadDependencies,
    build_diagnostic_read_service,
)
from intergrax.applications._shared.diagnostic_runtime_wiring import (
    build_terminal_execution_diagnostic_trigger,
    resolve_host_diagnostic_runtime_dependencies,
)
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_run_enums import CognitivePattern
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.llm_adapters._shared.adapter_response_builders import (
    build_adapter_response,
)
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.task.task import TaskContext
from typing import Optional, Sequence
from intergrax.contracts.decision_coordination import (
    DecisionCoordinationContribution,
    DecisionCoordinationShape,
)
from intergrax.contracts.execution_identity import (
    require_active_execution_identity,
    validate_task_id,
)
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.runtime.diagnostics.diagnostic_read_service import DiagnosticReadService
from intergrax.runtime.events.stores.memory_runtime_event_store import (
    InMemoryRuntimeEventStore,
)
from intergrax.runtime.execution.lineage.persistence import (
    InMemoryExecutionLineagePersistence,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.observability_wiring import wire_nexus_observability
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
from testing_support.agent_distribution.coordination_governance import (
    bound_governed_host_task,
)
from testing_support.agent_distribution.delegated_subtask_qualification_harness import (
    OCR_QUALIFICATION_PACKAGE_ID,
    OcrQualificationRequest,
    build_ocr_qualification_discovery_candidate,
)
from testing_support.agent_distribution.decision_coordination_qualification import (
    DecisionCoordinationExecutorFixture,
    accepted_decision,
    build_decision_coordination_executor_fixture,
    coordination_binding,
    decision_contribution,
    project_accepted_decision,
)
from testing_support.agent_platform_admin_harness import admin_test_principal


class _FakeEnvWiring:
    def __init__(self, document_store: InMemoryDocumentStore) -> None:
        self.build_context = _FakeBuildContext(document_store)


class _FakeBuildContext:
    def __init__(self, document_store: InMemoryDocumentStore) -> None:
        self.tool_wiring_context = _FakeToolWiringContext(document_store)


class _FakeToolWiringContext:
    def __init__(self, document_store: InMemoryDocumentStore) -> None:
        self.document_store = document_store


@dataclass(frozen=True, slots=True)
class Dg001CanonicalMultiAgentDiagnosticHarness:
    nexus_loop: NexusLoop
    runner: UnifiedTaskRunner
    lineage_persistence: InMemoryExecutionLineagePersistence
    runtime_event_store: InMemoryRuntimeEventStore
    diagnostic_dependencies: HostDiagnosticReadDependencies
    coordination_fixture: DecisionCoordinationExecutorFixture
    read_service: DiagnosticReadService
    tenant_id: str


class _Dg001StubLlmAdapter(LLMAdapter):
    provider = "dg001-stub"
    model = "dg001-stub"

    @property
    def context_window_tokens(self) -> int:
        return 128_000

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        _ = temperature, max_tokens, run_id, messages
        return build_adapter_response(content="dg001-stub")


class Dg001CoordinationRootAgent(ReflexAgent):
    """Root business delegate that executes CoordinationIntentExecutor under canonical root runtime."""

    contract_id = "dg001-multi-agent-root"
    capabilities = ("dg001.multi_agent.coordination",)
    agent_name = "DG-001 coordination root"
    agent_description = "Qualification root agent for multi-agent diagnostic P3"
    agent_version = "1.0.0"
    risk_level = AgentRiskLevel.LOW
    cognitive_pattern = CognitivePattern.REFLEX
    max_steps = 1

    def __init__(
        self,
        *,
        fixture: DecisionCoordinationExecutorFixture,
        contribution_lease_pairs: tuple[tuple[str, str], ...],
        shape: DecisionCoordinationShape,
        contributions: tuple[
            DecisionCoordinationContribution[OcrQualificationRequest],
            ...,
        ],
    ) -> None:
        self._fixture = fixture
        self._contribution_lease_pairs = contribution_lease_pairs
        self._accepted = accepted_decision(shape, contributions)
        self._harness = default_reference_harness()

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=self.contract_id,
            name=self.agent_name,
            description=self.agent_description,
            version=self.agent_version,
            capabilities=list(self.capabilities),
            skills=[],
            extra_tools=[],
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            validation_rules=["structured_output"],
            failure_modes=["coordination_failed"],
            risk_level=self.risk_level,
            max_steps=self.max_steps,
        )

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        capability = task_context.capability
        if capability in (None, "dg001.multi_agent.coordination"):
            return CapabilityMatchResult(
                matched=True,
                agent_id=self.contract_id,
                matched_capabilities=["dg001.multi_agent.coordination"],
                score=1.0,
                rationale="dg001 qualification root",
            )
        return CapabilityMatchResult(
            matched=False, rationale="capability not supported"
        )

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        return build_lab_agent_runtime_context(
            request=request,
            llm_adapter=_Dg001StubLlmAdapter(),
            harness=self._harness,
        )

    async def perceive(self, step_ctx: AgentStepContext) -> Observation:
        return Observation(summary=step_ctx.task_id or "dg001")

    async def reason(
        self,
        step_ctx: AgentStepContext,
        observation: Observation,
    ) -> ReasoningResult:
        return ReasoningResult(thought=observation.summary)

    async def act(
        self,
        step_ctx: AgentStepContext,
        reasoning: ReasoningResult,
    ) -> dict[str, object]:
        run_id, _ = require_active_execution_identity()
        resolved_task_id = ActiveTaskRegistry.peek_task_id_for_run(run_id)
        if resolved_task_id is None:
            raise RuntimeError("DG-001 root agent requires active task registration")
        task_scope = validate_task_id(resolved_task_id)
        intent = project_accepted_decision(self._accepted)
        harness = self._fixture.harness
        harness.task_scope_authority.task_scope_id = task_scope
        binding = coordination_binding(task_scope, self._contribution_lease_pairs)
        with bound_governed_host_task():
            await self._fixture.executor.execute(
                intent,
                binding=binding,
                principal=admin_test_principal(),
            )
        return {"summary": reasoning.thought, "status": "completed"}

    def evaluate(
        self,
        step_ctx: AgentStepContext,
        output: dict[str, object],
    ) -> AgentEvaluation:
        _ = step_ctx, output
        return AgentEvaluation(
            verdict=CognitiveEvaluation.COMPLETE, reason="dg001_root"
        )


def build_dg001_canonical_multi_agent_diagnostic_harness(
    *,
    tenant_id: str,
    specialist_delegate: object | None = None,
    fan_out: bool = False,
    shape: DecisionCoordinationShape = DecisionCoordinationShape.SINGLE,
    contributions: (
        tuple[DecisionCoordinationContribution[OcrQualificationRequest], ...] | None
    ) = None,
    contribution_lease_pairs: tuple[tuple[str, str], ...] | None = None,
) -> Dg001CanonicalMultiAgentDiagnosticHarness:
    resolved_contributions = contributions or (
        decision_contribution("contrib-a", document_ref="doc-single"),
    )
    resolved_pairs = contribution_lease_pairs or (("contrib-a", "lease-a"),)
    fixture = build_decision_coordination_executor_fixture(
        candidates=(
            build_ocr_qualification_discovery_candidate(
                OCR_QUALIFICATION_PACKAGE_ID,
                capability_ids=("document.ocr",),
            ),
        ),
        specialist_delegate=specialist_delegate,
        fan_out=fan_out,
    )
    document_store = InMemoryDocumentStore()
    runtime_store = InMemoryRuntimeEventStore()
    lineage = InMemoryExecutionLineagePersistence()
    stores = wire_nexus_observability(
        use_in_memory_trace=True,
        runtime_event_store=runtime_store,
    )
    runtime_deps = resolve_host_diagnostic_runtime_dependencies(
        env_wiring=_FakeEnvWiring(document_store),
        observability=stores,
    )
    if runtime_deps is None:
        raise RuntimeError("DG-001 harness requires diagnostic runtime dependencies")
    read_deps = replace(runtime_deps, execution_lineage_reader=lineage)
    trigger = build_terminal_execution_diagnostic_trigger(read_deps)
    registry = AgentRegistry()
    registry.register(
        Dg001CoordinationRootAgent(
            fixture=fixture,
            contribution_lease_pairs=resolved_pairs,
            shape=shape,
            contributions=resolved_contributions,
        ),
    )
    loop = NexusLoop(
        registry,
        trace_store=stores.trace_store,
        runtime_event_store=runtime_store,
        execution_lineage_persistence=lineage,
    )
    loop.attach_terminal_diagnostic_trigger(trigger)
    return Dg001CanonicalMultiAgentDiagnosticHarness(
        nexus_loop=loop,
        runner=UnifiedTaskRunner(loop),
        lineage_persistence=lineage,
        runtime_event_store=runtime_store,
        diagnostic_dependencies=read_deps,
        coordination_fixture=fixture,
        read_service=build_diagnostic_read_service(read_deps),
        tenant_id=tenant_id,
    )
