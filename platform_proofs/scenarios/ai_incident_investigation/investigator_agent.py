# © Artur Czarnecki. All rights reserved.

"""Investigator agent — lifecycle via UAEP; model-owned reasoning with bounded tool loop."""

from __future__ import annotations

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.diagnostics.investigation_contracts import IncidentInvestigationInput
from intergrax.runtime.nexus.context.metadata_keys import PRIOR_AGENT_OUTPUTS_KEY
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from platform_proofs.scenarios.ai_incident_investigation.evidence_gathering import (
    gather_incident_evidence,
)
from platform_proofs.scenarios.ai_incident_investigation.incident_reasoning import (
    build_investigation_summary,
    completion_mode_from_proposal,
    convert_proposal_to_pending_claims,
    extract_prior_investigation_state,
    propose_incident_reasoning,
    emit_reasoning_observability,
)
from platform_proofs.scenarios.ai_incident_investigation.incident_scope import IncidentScope
from platform_proofs.scenarios.ai_incident_investigation.runtime_composition import (
    ScenarioRuntimeComposition,
    build_agent_runtime_context,
)
from platform_proofs.scenarios.ai_incident_investigation.scenario_contract import (  # noqa: F401
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
from platform_proofs.scenarios.ai_incident_investigation.tools import SCENARIO_TOOL_IDS

INVESTIGATOR_AGENT_ID = "incident_investigator"
INVESTIGATOR_CAPABILITY = "incident_investigation.investigate"


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


def _is_revision(ctx: RuntimeExecutionContext) -> bool:
    if (ctx.request.metadata or {}).get("critic_feedback"):
        return True
    raw_feedback = ctx.metadata.get("critic_feedback")
    return isinstance(raw_feedback, list) and bool(raw_feedback)


def _prior_metadata(ctx: RuntimeExecutionContext) -> dict[str, object]:
    request_meta = dict(ctx.request.metadata or {}) if ctx.request is not None else {}
    if PRIOR_AGENT_OUTPUTS_KEY in request_meta:
        return request_meta
    prior = ctx.metadata.get(PRIOR_AGENT_OUTPUTS_KEY)
    if isinstance(prior, dict):
        return {PRIOR_AGENT_OUTPUTS_KEY: prior}
    return request_meta


class IncidentInvestigatorAgent(Agent):
    def __init__(
        self,
        registry: object,
        station_id: str,
        runtime_composition: ScenarioRuntimeComposition,
        incident_scope: IncidentScope,
        evidence_store: object | None = None,
        investigation_input: IncidentInvestigationInput | None = None,
    ) -> None:
        from intergrax.tools.registry import ToolRegistry
        from platform_proofs.scenarios.ai_incident_investigation.tools import ScenarioEvidenceStore

        if not isinstance(registry, ToolRegistry):
            raise TypeError("registry must be ToolRegistry")
        self._registry = registry
        self._station_id = station_id
        self._runtime_composition = runtime_composition
        self._incident_scope = incident_scope
        self._evidence_store = (
            evidence_store if isinstance(evidence_store, ScenarioEvidenceStore) else None
        )
        self._investigation_input = investigation_input

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
        is_revision = _is_revision(ctx)
        runtime_state = ctx.metadata.get("runtime_state")
        if not isinstance(runtime_state, RuntimeState):
            raise RuntimeError("runtime_state_not_bound_for_tool_runtime")

        node_id = str((ctx.request.metadata or {}).get("graph_node_id") or ctx.node_id or "")
        prior_state = extract_prior_investigation_state(
            _prior_metadata(ctx),
            node_id=node_id or None,
        )
        critic_feedback = _extract_critic_feedback(ctx, is_revision)

        gathering = gather_incident_evidence(
            runtime_state=runtime_state,
            registry=self._registry,
            scope=self._incident_scope,
            is_revision=is_revision,
            critic_feedback=critic_feedback,
            prior_evidence=prior_state.evidence_nodes,
            evidence_store=self._evidence_store,
        )

        evidence_nodes = list(gathering.evidence_nodes)
        proposal = propose_incident_reasoning(
            runtime_state=runtime_state,
            evidence_nodes=evidence_nodes,
            prior_state=prior_state,
            critic_feedback=critic_feedback,
            is_revision=is_revision,
            investigation_input=self._investigation_input,
        )
        pending_conversion = convert_proposal_to_pending_claims(
            proposal,
            prior_claim_set=prior_state.claim_set,
            prior_bindings=prior_state.claim_hypothesis_bindings,
            critic_feedback=critic_feedback,
        )
        pending_claim_set = pending_conversion.claim_set
        claim_bindings = pending_conversion.bindings
        completion_mode = completion_mode_from_proposal(proposal)
        emit_reasoning_observability(
            runtime_state=runtime_state,
            proposal=proposal,
            claim_set=pending_claim_set,
            bindings=claim_bindings,
            is_revision=is_revision,
            critic_feedback=critic_feedback,
        )

        summary = build_investigation_summary(proposal, is_revision=is_revision)
        domain_payload = {
            "claim_set": pending_claim_set.model_dump(mode="json"),
            "claim_hypothesis_bindings": [
                binding.model_dump(mode="json") for binding in claim_bindings
            ],
            "evidence_nodes": evidence_nodes,
            "reasoning_proposal": proposal.model_dump(mode="json"),
            "active_hypothesis": proposal.preferred_hypothesis_id,
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
