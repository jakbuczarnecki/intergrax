# © Artur Czarnecki. All rights reserved.

"""Investigator agent — lifecycle via UAEP; autonomous evidence via bounded tool loop."""

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
from intergrax.contracts.capability import CapabilityMatchResult
from platform_proofs.scenarios.ai_incident_investigation.domain_reasoning import (
    IncidentAssessment,
    RationaleCode,
    derive_hypothesis_dispositions,
    observations_from_evidence_nodes,
)
from platform_proofs.scenarios.ai_incident_investigation.evidence_gathering import (
    gather_incident_evidence,
)
from platform_proofs.scenarios.ai_incident_investigation.incident_scope import IncidentScope
from platform_proofs.scenarios.ai_incident_investigation.tools import SCENARIO_TOOL_IDS
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


def _extract_critic_feedback(ctx: RuntimeExecutionContext, is_revision: bool) -> list[str]:
    if not is_revision:
        return []
    request_feedback = (ctx.request.metadata or {}).get("critic_feedback")
    if isinstance(request_feedback, list):
        return [str(item) for item in request_feedback]
    raw_feedback = ctx.metadata.get("critic_feedback")
    if isinstance(raw_feedback, list):
        return [str(item) for item in raw_feedback]
    return []


class IncidentInvestigatorAgent(Agent):
    def __init__(
        self,
        registry: object,
        station_id: str,
        runtime_composition: ScenarioRuntimeComposition,
        incident_scope: IncidentScope,
    ) -> None:
        from intergrax.tools.registry import ToolRegistry

        if not isinstance(registry, ToolRegistry):
            raise TypeError("registry must be ToolRegistry")
        self._registry = registry
        self._station_id = station_id
        self._runtime_composition = runtime_composition
        self._incident_scope = incident_scope

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

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        _ = step
        is_revision = bool((ctx.request.metadata or {}).get("critic_feedback"))
        if not is_revision:
            raw_feedback = ctx.metadata.get("critic_feedback")
            if isinstance(raw_feedback, list) and raw_feedback:
                is_revision = True
        runtime_state = ctx.metadata.get("runtime_state")
        if not isinstance(runtime_state, RuntimeState):
            raise RuntimeError("runtime_state_not_bound_for_tool_runtime")

        gathering = gather_incident_evidence(
            runtime_state=runtime_state,
            registry=self._registry,
            scope=self._incident_scope,
            is_revision=is_revision,
            critic_feedback=_extract_critic_feedback(ctx, is_revision),
        )

        evidence_nodes = list(gathering.evidence_nodes)
        observations = observations_from_evidence_nodes(evidence_nodes, INCIDENT_EVIDENCE_IDS)
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
            "tool_invocations": gathering.tool_invocations,
            "revision_pass": is_revision,
            "initial_evidence_ids": list(gathering.initial_evidence_ids),
            "evidence_gathering_stop_reason": gathering.stop_reason,
            "tool_execution_order": list(gathering.tool_execution_order),
            "planner_decisions": list(gathering.planner_decisions),
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
