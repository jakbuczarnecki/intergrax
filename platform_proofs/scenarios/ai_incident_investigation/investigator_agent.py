# © Artur Czarnecki. All rights reserved.

"""Investigator agent — lifecycle via UAEP; tools via BoundToolGateway / ToolRuntime."""

from __future__ import annotations

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.evidence_claims import (
    ClaimResolution,
    EvidenceBackedClaim,
    EvidenceClaimSet,
    validate_claim_kind,
    validate_evidence_claim_id,
    validate_evidence_reference_id,
)
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.tools.tool_runtime import ToolInvocationPlan, ToolRuntime
from unittest.mock import MagicMock

from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.tools.registry import ToolRegistry
from platform_proofs.scenarios.ai_incident_investigation.fixtures import HypothesisId
from platform_proofs.scenarios.ai_incident_investigation.tools import (
    default_line_window_input,
    default_telemetry_input,
    SCENARIO_TOOL_IDS,
    TOOL_TELEMETRY_READ,
    TOOL_THROUGHPUT_READ,
    TOOL_WORKLOAD_READ,
)
from platform_proofs.scenarios.ai_incident_investigation.validation import DIAGNOSIS_CLAIM_KIND
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

INVESTIGATOR_AGENT_ID = "incident_investigator"
INVESTIGATOR_CAPABILITY = "incident_investigation.investigate"

INITIAL_CLAIM_ID = validate_evidence_claim_id("eclaim_a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1")
REVISED_CLAIM_ID = validate_evidence_claim_id("eclaim_b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2")
WORKLOAD_EVIDENCE_ID = validate_evidence_reference_id("evidence.workload.line4.incident_window")
THROUGHPUT_EVIDENCE_ID = validate_evidence_reference_id("evidence.throughput.line4.incident_window")
TELEMETRY_EVIDENCE_ID = validate_evidence_reference_id(
    "evidence.telemetry.complex_assembly_station.incident_window"
)
DIAGNOSIS_KIND = validate_claim_kind(DIAGNOSIS_CLAIM_KIND)


class IncidentInvestigatorAgent(Agent):
    def __init__(self, registry: ToolRegistry, station_id: str) -> None:
        self._registry = registry
        self._station_id = station_id

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=INVESTIGATOR_AGENT_ID,
            name=INVESTIGATOR_AGENT_ID,
            description="Incident investigator — platform-native skeleton",
            capabilities=[INVESTIGATOR_CAPABILITY],
            allowed_tools=list(SCENARIO_TOOL_IDS),
        )

    def can_handle(self, task_context: object) -> CapabilityMatchResult:
        capability = getattr(task_context, "capability", None)
        if capability == INVESTIGATOR_CAPABILITY:
            return CapabilityMatchResult(
                matched=True,
                agent_id=INVESTIGATOR_AGENT_ID,
                matched_capabilities=[INVESTIGATOR_CAPABILITY],
                score=1.0,
            )
        return CapabilityMatchResult(matched=False)

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        invoker = RuntimeToolInvoker(
            registry=self._registry,
            executor=RegistryToolExecutor(self._registry),
        )
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(fixed_text="investigate"),
            enable_rag=False,
            production_mode=False,
            tenant_id=request.tenant_id,
            tool_invoker=invoker,
            tool_registry=self._registry,
            tools_mode="catalog",
        )
        from intergrax.runtime.events.event_bus import RuntimeEventBus

        config.runtime_event_bus = RuntimeEventBus()
        return RuntimeContext(
            config=config,
            session_manager=build_in_memory_session_manager(),
            prompt_registry=MagicMock(),
        )

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        _ = context
        return [
            AgentStep(
                step_id="investigate",
                step_name="investigate",
                step_index=0,
                trace_label=INVESTIGATOR_CAPABILITY,
                allowed_tools=list(SCENARIO_TOOL_IDS),
            )
        ]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        is_revision = bool((ctx.request.metadata or {}).get("critic_feedback"))
        if not is_revision:
            raw_feedback = ctx.metadata.get("critic_feedback")
            if isinstance(raw_feedback, list) and raw_feedback:
                is_revision = True
        runtime_state = ctx.metadata.get("runtime_state")
        if not isinstance(runtime_state, RuntimeState):
            raise RuntimeError("runtime_state_not_bound_for_tool_runtime")

        tool_invocations = 0
        line_input = default_line_window_input()

        await ToolRuntime.invoke(
            state=runtime_state,
            plan=ToolInvocationPlan(
                tool_ids=(TOOL_WORKLOAD_READ,),
                tool_inputs={TOOL_WORKLOAD_READ: line_input},
            ),
            trace_step=step.step_id,
            allowed_tools=SCENARIO_TOOL_IDS,
        )
        tool_invocations += 1
        workload_trace = runtime_state.tool_traces[-1]
        if not workload_trace.success:
            raise RuntimeError(f"workload tool failed: {workload_trace.error_message}")

        await ToolRuntime.invoke(
            state=runtime_state,
            plan=ToolInvocationPlan(
                tool_ids=(TOOL_THROUGHPUT_READ,),
                tool_inputs={TOOL_THROUGHPUT_READ: line_input},
            ),
            trace_step=step.step_id,
            allowed_tools=SCENARIO_TOOL_IDS,
        )
        tool_invocations += 1
        throughput_trace = runtime_state.tool_traces[-1]
        if not throughput_trace.success:
            raise RuntimeError(f"throughput tool failed: {throughput_trace.error_message}")

        evidence_nodes = [
            {
                "evidence_id": str(WORKLOAD_EVIDENCE_ID),
                "kind": "tool_result",
                "label": "workload observation",
                "payload": workload_trace.output_preview,
            },
            {
                "evidence_id": str(THROUGHPUT_EVIDENCE_ID),
                "kind": "tool_result",
                "label": "throughput observation",
                "payload": throughput_trace.output_preview,
            },
        ]

        if is_revision:
            telemetry_input = default_telemetry_input(self._station_id)
            await ToolRuntime.invoke(
                state=runtime_state,
                plan=ToolInvocationPlan(
                    tool_ids=(TOOL_TELEMETRY_READ,),
                    tool_inputs={TOOL_TELEMETRY_READ: telemetry_input},
                ),
                trace_step=step.step_id,
                allowed_tools=SCENARIO_TOOL_IDS,
            )
            tool_invocations += 1
            telemetry_trace = runtime_state.tool_traces[-1]
            if not telemetry_trace.success:
                raise RuntimeError(f"telemetry tool failed: {telemetry_trace.error_message}")
            evidence_nodes.append(
                {
                    "evidence_id": str(TELEMETRY_EVIDENCE_ID),
                    "kind": "tool_result",
                    "label": "station telemetry observation",
                    "payload": telemetry_trace.output_preview,
                }
            )
            initial_claim = EvidenceBackedClaim(
                claim_id=INITIAL_CLAIM_ID,
                statement=(
                    "Production workload on Line 4 increased during the incident window "
                    "while throughput declined — overload hypothesis H1."
                ),
                claim_kind=DIAGNOSIS_KIND,
                supporting_evidence_ids=(WORKLOAD_EVIDENCE_ID, THROUGHPUT_EVIDENCE_ID),
                resolution=ClaimResolution.SUPERSEDED,
            )
            revised_claim = EvidenceBackedClaim(
                claim_id=REVISED_CLAIM_ID,
                statement=(
                    "Intermittent station signal degradation on the complex-assembly step "
                    "correlates with throughput loss; workload growth is a contributing factor, "
                    "not the initiating cause — bounded H3 diagnosis."
                ),
                claim_kind=DIAGNOSIS_KIND,
                supporting_evidence_ids=(
                    WORKLOAD_EVIDENCE_ID,
                    THROUGHPUT_EVIDENCE_ID,
                    TELEMETRY_EVIDENCE_ID,
                ),
                resolution=ClaimResolution.SUPPORTED,
                supersedes_claim_id=INITIAL_CLAIM_ID,
            )
            claim_set = EvidenceClaimSet(claims=(initial_claim, revised_claim), challenges=())
            active_hypothesis = HypothesisId.H3
            summary = (
                "revised: bounded equipment-process degradation diagnosis supported by telemetry"
            )
        else:
            draft_claim = EvidenceBackedClaim(
                claim_id=INITIAL_CLAIM_ID,
                statement=(
                    "Sustained production overload from workload growth caused Line 4 "
                    "target attainment degradation — hypothesis H1."
                ),
                claim_kind=DIAGNOSIS_KIND,
                supporting_evidence_ids=(WORKLOAD_EVIDENCE_ID, THROUGHPUT_EVIDENCE_ID),
                resolution=ClaimResolution.PENDING,
            )
            claim_set = EvidenceClaimSet(claims=(draft_claim,), challenges=())
            active_hypothesis = HypothesisId.H1
            summary = "draft: workload overload candidate diagnosis hypothesis H1"

        domain_payload = {
            "claim_set": claim_set.model_dump(mode="json"),
            "evidence_nodes": evidence_nodes,
            "active_hypothesis": str(active_hypothesis),
            "tool_invocations": tool_invocations,
            "revision_pass": is_revision,
        }
        return StepOutput(
            step_id=step.step_id,
            summary=summary,
            data={"domain_summary": domain_payload},
        )

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output, ctx
        return AgentDecision(
            type=AgentDecisionType.COMPLETE,
            reason="incident investigation step complete",
        )
