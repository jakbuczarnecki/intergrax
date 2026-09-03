# © Artur Czarnecki. All rights reserved.

"""ACP decision gateway — Tier-2 calls Decision flow gate via harness metadata (DS-MIG-01)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.acp_metadata_keys import AcpRunContextKey
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.execution_identity import require_active_execution_identity
from intergrax.runtime.decision_flow import (
    CanonicalDecisionFlowGate,
    DecisionFlowGate,
    DecisionFlowHostAction,
    DecisionFlowScope,
)
from intergrax.runtime.decision_flow_host import (
    agent_execution_decision_context,
    agent_execution_identity_seed,
    build_agent_execution_flow_request,
    evaluate_agent_execution_flow,
)


@dataclass(frozen=True, slots=True)
class ReflectionDecisionOutcome:
    """Normalized Decision flow result for reflection pattern authors."""

    passed: bool
    host_action: DecisionFlowHostAction
    summary: str


def resolve_decision_flow_gate(
    step_ctx: AgentStepContext,
) -> DecisionFlowGate[AgentExecutionResult] | None:
    raw = step_ctx.metadata.get(AcpRunContextKey.DECISION_FLOW_GATE)
    if isinstance(raw, CanonicalDecisionFlowGate):
        return raw
    return None


def _resolve_tenant_id(step_ctx: AgentStepContext) -> str:
    tenant = step_ctx.metadata.get(AcpRunContextKey.TENANT_ID)
    if isinstance(tenant, str) and tenant:
        return tenant
    org = step_ctx.metadata.get(AcpRunContextKey.ORGANIZATIONAL)
    if isinstance(org, dict):
        nested = org.get("tenant_id")
        if isinstance(nested, str) and nested:
            return nested
    return "default"


async def verify_reflection_draft_with_decision(
    step_ctx: AgentStepContext,
    *,
    contract: AgentContract,
    draft: str,
    step_id: str = "reflection_critique",
) -> ReflectionDecisionOutcome | None:
    """Run Decision flow verification on a reflection draft when gate is wired."""
    gate = resolve_decision_flow_gate(step_ctx)
    if gate is None or not draft.strip():
        return None
    if not gate.supports_scope(DecisionFlowScope.UAEP_STEP):
        return None
    execution = AgentExecutionResult(
        agent_id=contract.id,
        run_id=step_ctx.run_id,
        status=AgentExecutionStatus.COMPLETED,
        summary=draft,
        structured_data={"draft": draft, "phase": "critique"},
    )
    active_run_id, active_attempt_id = require_active_execution_identity()
    decision_context = agent_execution_decision_context(
        task_id=step_ctx.task_id,
        run_id=active_run_id,
        attempt_id=active_attempt_id,
        tenant_id=_resolve_tenant_id(step_ctx),
    )
    identity_seed = agent_execution_identity_seed(
        context=decision_context,
        namespace="uaep.reflection",
        subject=step_id,
    )
    flow_request = build_agent_execution_flow_request(
        execution=execution,
        identity_seed=identity_seed,
        flow_scope=DecisionFlowScope.UAEP_STEP,
    )
    flow_result = await evaluate_agent_execution_flow(gate, flow_request)
    summary = flow_result.authority_reason or draft[:240]
    return ReflectionDecisionOutcome(
        passed=flow_result.host_action is DecisionFlowHostAction.CONTINUE,
        host_action=flow_result.host_action,
        summary=summary,
    )


def decision_flow_gate_attached(step_ctx: AgentStepContext) -> bool:
    return resolve_decision_flow_gate(step_ctx) is not None
