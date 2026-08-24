# © Artur Czarnecki. All rights reserved.

"""Investigator agent — lifecycle via UAEP; tools via BoundToolGateway / ToolRuntime."""

from __future__ import annotations

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.evidence_claims import (
    ClaimResolution,
    EvidenceBackedClaim,
    EvidenceClaimSet,
    validate_evidence_reference_id,
)
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from platform_proofs.scenarios.ai_incident_investigation.scenario_contract import (
    COMPARISON_EVIDENCE_ID,
    COMPLETION_SUPPORTED_DIAGNOSIS,
    COMPLETION_UNRESOLVED,
    DIAGNOSIS_KIND,
    H2_CLAIM_ID,
    H3_CLAIM_ID,
    INCIDENT_EVIDENCE_IDS,
    INITIAL_CLAIM_ID,
    REVISED_CLAIM_ID,
    STAFFING_ATTENDANCE_EVIDENCE_ID,
    STAFFING_PRELIMINARY_EVIDENCE_ID,
    TELEMETRY_EVIDENCE_ID,
    THROUGHPUT_EVIDENCE_ID,
    WORKLOAD_EVIDENCE_ID,
)
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.contracts.tool_request import ToolRequest, ToolResponseStatus
from intergrax.runtime.nexus.tools.tool_runtime import ToolRuntime
from intergrax.tools.registry import ToolRegistry
from intergrax.contracts.capability import CapabilityMatchResult
from platform_proofs.scenarios.ai_incident_investigation.domain_reasoning import (
    IncidentAssessment,
    IncidentObservations,
    RationaleCode,
    derive_hypothesis_dispositions,
    parse_comparison_payload,
    parse_staffing_attendance_payload,
    parse_staffing_schedule_payload,
    parse_telemetry_payload,
    parse_throughput_payload,
    parse_workload_payload,
)
from platform_proofs.scenarios.ai_incident_investigation.tools import (
    default_comparison_input,
    default_line_window_input,
    default_staffing_input,
    default_telemetry_input,
    SCENARIO_TOOL_IDS,
    TOOL_COMPARISON_READ,
    TOOL_STAFFING_ATTENDANCE_READ,
    TOOL_STAFFING_SCHEDULE_READ,
    TOOL_TELEMETRY_READ,
    TOOL_THROUGHPUT_READ,
    TOOL_WORKLOAD_READ,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures import HypothesisId
from platform_proofs.scenarios.ai_incident_investigation.runtime_composition import (
    ScenarioRuntimeComposition,
    build_agent_runtime_context,
)

INVESTIGATOR_AGENT_ID = "incident_investigator"
INVESTIGATOR_CAPABILITY = "incident_investigation.investigate"


def _build_claims_from_assessment(assessment: IncidentAssessment) -> EvidenceClaimSet:
    h1 = assessment.h1
    h2 = assessment.h2
    h3 = assessment.h3

    h1_statement = (
        "Production workload on Line 4 increased during the incident window "
        "while throughput declined — overload hypothesis H1."
        if h1.disposition is not ClaimResolution.PENDING
        else (
            "Sustained production overload from workload growth caused Line 4 "
            "target attainment degradation — hypothesis H1."
        )
    )
    initial_claim = EvidenceBackedClaim(
        claim_id=INITIAL_CLAIM_ID,
        statement=h1_statement,
        claim_kind=DIAGNOSIS_KIND,
        supporting_evidence_ids=tuple(
            validate_evidence_reference_id(eid) for eid in h1.supporting_evidence_ids
        ),
        contradicting_evidence_ids=tuple(
            validate_evidence_reference_id(eid) for eid in h1.contradicting_evidence_ids
        ),
        resolution=h1.disposition,
    )

    h2_statement = (
        "Understaffing on the affected shift is not supported as initiating cause: "
        "preliminary roster export conflicts with confirmed attendance for the "
        "incident window — hypothesis H2 rejected."
        if h2.disposition is ClaimResolution.REJECTED
        else (
            "Understaffing on the affected shift is supported by confirmed attendance "
            "below required headcount — hypothesis H2."
            if h2.disposition is ClaimResolution.SUPPORTED
            else "Understaffing hypothesis H2 pending further staffing evidence."
        )
    )
    h2_claim = EvidenceBackedClaim(
        claim_id=H2_CLAIM_ID,
        statement=h2_statement,
        claim_kind=DIAGNOSIS_KIND,
        supporting_evidence_ids=tuple(
            validate_evidence_reference_id(eid) for eid in h2.supporting_evidence_ids
        ),
        contradicting_evidence_ids=tuple(
            validate_evidence_reference_id(eid) for eid in h2.contradicting_evidence_ids
        ),
        resolution=h2.disposition,
    )

    claims: list[EvidenceBackedClaim] = [initial_claim, h2_claim]

    if h3.disposition is ClaimResolution.SUPPORTED:
        revised_claim = EvidenceBackedClaim(
            claim_id=REVISED_CLAIM_ID,
            statement=(
                "Intermittent station signal degradation on the complex-assembly step "
                "is the best-supported initiating cause; comparison evidence shows "
                "similar elevated workload elsewhere without comparable degradation; "
                "workload growth plausibly amplified impact — bounded H3 diagnosis."
            ),
            claim_kind=DIAGNOSIS_KIND,
            supporting_evidence_ids=tuple(
                validate_evidence_reference_id(eid) for eid in h3.supporting_evidence_ids
            ),
            contradicting_evidence_ids=tuple(
                validate_evidence_reference_id(eid) for eid in h3.contradicting_evidence_ids
            ),
            resolution=ClaimResolution.SUPPORTED,
            supersedes_claim_id=INITIAL_CLAIM_ID,
        )
        claims.append(revised_claim)
    elif (
        h3.disposition is ClaimResolution.INSUFFICIENT_EVIDENCE
        and h3.rationale_code is RationaleCode.H3_INSUFFICIENT_TELEMETRY_UNAVAILABLE
    ):
        h3_claim = EvidenceBackedClaim(
            claim_id=H3_CLAIM_ID,
            statement=(
                "Equipment-process degradation hypothesis H3 cannot be accepted: "
                "decisive station telemetry for the incident window is unavailable."
            ),
            claim_kind=DIAGNOSIS_KIND,
            supporting_evidence_ids=(),
            contradicting_evidence_ids=(),
            resolution=ClaimResolution.INSUFFICIENT_EVIDENCE,
        )
        claims.append(h3_claim)

    return EvidenceClaimSet(claims=tuple(claims), challenges=())


class IncidentInvestigatorAgent(Agent):
    def __init__(
        self,
        registry: ToolRegistry,
        station_id: str,
        runtime_composition: ScenarioRuntimeComposition,
    ) -> None:
        self._registry = registry
        self._station_id = station_id
        self._runtime_composition = runtime_composition

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=INVESTIGATOR_AGENT_ID,
            name=INVESTIGATOR_AGENT_ID,
            description="Incident investigator — platform-native scenario",
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
        return build_agent_runtime_context(request, self._runtime_composition)

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

    async def _invoke_tool(
        self,
        *,
        runtime_state: RuntimeState,
        step: AgentStep,
        tool_id: str,
        tool_input: dict[str, str],
    ) -> dict[str, object]:
        response = await ToolRuntime.invoke_request(
            state=runtime_state,
            request=ToolRequest(
                tool_name=tool_id,
                agent_id=INVESTIGATOR_AGENT_ID,
                step_id=step.step_id,
                input=tool_input,
            ),
            allowed_tools=SCENARIO_TOOL_IDS,
            trace_step=step.step_id,
        )
        if response.status is not ToolResponseStatus.SUCCESS:
            raise RuntimeError(
                f"{tool_id} failed: status={response.status.value} error={response.error}"
            )
        if response.output is None:
            raise RuntimeError(f"{tool_id} succeeded but output is missing")
        return response.output

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
        staffing_input = default_staffing_input()

        workload_output = await self._invoke_tool(
            runtime_state=runtime_state,
            step=step,
            tool_id=TOOL_WORKLOAD_READ,
            tool_input=line_input,
        )
        tool_invocations += 1

        throughput_output = await self._invoke_tool(
            runtime_state=runtime_state,
            step=step,
            tool_id=TOOL_THROUGHPUT_READ,
            tool_input=line_input,
        )
        tool_invocations += 1

        staffing_output = await self._invoke_tool(
            runtime_state=runtime_state,
            step=step,
            tool_id=TOOL_STAFFING_SCHEDULE_READ,
            tool_input=staffing_input,
        )
        tool_invocations += 1

        evidence_nodes: list[dict[str, object]] = [
            {
                "evidence_id": str(WORKLOAD_EVIDENCE_ID),
                "kind": "tool_result",
                "label": "workload observation",
                "payload": workload_output,
            },
            {
                "evidence_id": str(THROUGHPUT_EVIDENCE_ID),
                "kind": "tool_result",
                "label": "throughput observation",
                "payload": throughput_output,
            },
            {
                "evidence_id": str(STAFFING_PRELIMINARY_EVIDENCE_ID),
                "kind": "tool_result",
                "label": "staffing schedule observation",
                "payload": staffing_output,
            },
        ]

        observations = IncidentObservations(
            workload=parse_workload_payload(workload_output),
            throughput=parse_throughput_payload(throughput_output),
            staffing_schedule=parse_staffing_schedule_payload(staffing_output),
        )

        if is_revision:
            comparison_output = await self._invoke_tool(
                runtime_state=runtime_state,
                step=step,
                tool_id=TOOL_COMPARISON_READ,
                tool_input=default_comparison_input(),
            )
            tool_invocations += 1

            attendance_output = await self._invoke_tool(
                runtime_state=runtime_state,
                step=step,
                tool_id=TOOL_STAFFING_ATTENDANCE_READ,
                tool_input=staffing_input,
            )
            tool_invocations += 1

            telemetry_output = await self._invoke_tool(
                runtime_state=runtime_state,
                step=step,
                tool_id=TOOL_TELEMETRY_READ,
                tool_input=default_telemetry_input(self._station_id),
            )
            tool_invocations += 1

            evidence_nodes.extend(
                [
                    {
                        "evidence_id": str(COMPARISON_EVIDENCE_ID),
                        "kind": "tool_result",
                        "label": "comparison line observation",
                        "payload": comparison_output,
                    },
                    {
                        "evidence_id": str(STAFFING_ATTENDANCE_EVIDENCE_ID),
                        "kind": "tool_result",
                        "label": "staffing attendance observation",
                        "payload": attendance_output,
                    },
                    {
                        "evidence_id": str(TELEMETRY_EVIDENCE_ID),
                        "kind": "tool_result",
                        "label": "station telemetry observation",
                        "payload": telemetry_output,
                    },
                ]
            )

            observations = IncidentObservations(
                workload=observations.workload,
                throughput=observations.throughput,
                staffing_schedule=observations.staffing_schedule,
                staffing_attendance=parse_staffing_attendance_payload(attendance_output),
                comparison=parse_comparison_payload(comparison_output),
                telemetry=parse_telemetry_payload(telemetry_output),
            )

        assessment = derive_hypothesis_dispositions(observations, INCIDENT_EVIDENCE_IDS)
        claim_set = _build_claims_from_assessment(assessment)
        active_hypothesis = assessment.active_hypothesis
        summary = assessment.summary
        completion_mode = COMPLETION_SUPPORTED_DIAGNOSIS
        if is_revision and assessment.h3.disposition is ClaimResolution.SUPPORTED:
            summary = f"revised: {summary}"
        elif (
            is_revision
            and assessment.h3.rationale_code
            is RationaleCode.H3_INSUFFICIENT_TELEMETRY_UNAVAILABLE
            and assessment.h3.disposition is ClaimResolution.INSUFFICIENT_EVIDENCE
        ):
            completion_mode = COMPLETION_UNRESOLVED

        domain_payload = {
            "claim_set": claim_set.model_dump(mode="json"),
            "evidence_nodes": evidence_nodes,
            "active_hypothesis": str(active_hypothesis),
            "completion_mode": completion_mode,
            "tool_invocations": tool_invocations,
            "revision_pass": is_revision,
            "initial_evidence_ids": [
                str(WORKLOAD_EVIDENCE_ID),
                str(THROUGHPUT_EVIDENCE_ID),
                str(STAFFING_PRELIMINARY_EVIDENCE_ID),
            ],
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
