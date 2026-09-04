# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Host adapters for Decision flow gate integration (DS-MIG-01)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionScope,
    validate_decision_tenant_id,
)
from intergrax.contracts.decision_record import (
    DecisionArtifactKind,
    validate_decision_artifact_kind,
)
from intergrax.contracts.decision_verification_stage import (
    VerificationStageRegistration,
    verification_stage_registry,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.decision_flow import (
    DecisionFlowGate,
    DecisionFlowHostAction,
    DecisionFlowIdentitySeed,
    DecisionFlowRequest,
    DecisionFlowResult,
    DecisionFlowScope,
)
from intergrax.runtime.decision_verification import VerificationPipeline
from intergrax.runtime.decision_verification_stages.structural import (
    STRUCTURAL_VERIFICATION_STAGE_KIND,
    AgentExecutionStructuralValidator,
    StructuralVerificationStage,
)

_AGENT_EXECUTION_ARTIFACT_KIND = validate_decision_artifact_kind("agent.execution.result")


@dataclass(frozen=True, slots=True)
class AgentExecutionDecisionContext:
    """Neutral execution context for Graph and UAEP decision identity seeds."""

    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    tenant_id: str
    execution_id: ExecutionId | None = None


def agent_execution_decision_context(
    *,
    task_id: TaskId | str,
    run_id: RunId | str,
    attempt_id: AttemptId | str,
    tenant_id: str,
    execution_id: ExecutionId | str | None = None,
) -> AgentExecutionDecisionContext:
    """Build one validated agent-execution decision context."""
    resolved_execution_id = (
        validate_execution_id(execution_id)
        if execution_id is not None
        else None
    )
    return AgentExecutionDecisionContext(
        task_id=validate_task_id(task_id),
        run_id=validate_run_id(run_id),
        attempt_id=validate_attempt_id(attempt_id),
        tenant_id=validate_decision_tenant_id(tenant_id),
        execution_id=resolved_execution_id,
    )


def agent_execution_identity_seed(
    *,
    context: AgentExecutionDecisionContext,
    namespace: str,
    subject: str,
) -> DecisionFlowIdentitySeed:
    """Build one identity seed for agent execution artifacts."""
    if type(context) is not AgentExecutionDecisionContext:
        raise TypeError("context must be AgentExecutionDecisionContext")
    execution = DecisionExecutionLineage(
        task_id=context.task_id,
        run_id=context.run_id,
        attempt_id=context.attempt_id,
        execution_id=context.execution_id,
    )
    return DecisionFlowIdentitySeed(
        scope=DecisionScope(namespace=namespace, subject=subject),
        tenant_id=context.tenant_id,
        execution=execution,
    )


def build_agent_execution_verification_pipeline(
    *,
    contract: AgentContract,
    capability: str | None = None,
    plan_criteria: tuple[str, ...] = (),
) -> VerificationPipeline[AgentExecutionResult]:
    """Compose structural-only verification for agent execution artifacts."""
    if type(contract) is not AgentContract:
        raise TypeError("contract must be AgentContract")
    structural_stage = StructuralVerificationStage(
        validators=(
            AgentExecutionStructuralValidator(
                contract=contract,
                capability=capability,
                plan_criteria=plan_criteria,
            ),
        ),
    )
    registry = verification_stage_registry(
        (
            VerificationStageRegistration(
                kind=STRUCTURAL_VERIFICATION_STAGE_KIND,
                stage=structural_stage,
                required=True,
            ),
        ),
    )
    return VerificationPipeline(registry=registry)


def build_agent_execution_flow_request(
    *,
    execution: AgentExecutionResult,
    identity_seed: DecisionFlowIdentitySeed,
    flow_scope: DecisionFlowScope,
    artifact_kind: DecisionArtifactKind | None = None,
) -> DecisionFlowRequest[AgentExecutionResult]:
    """Build one decision-flow request for an agent execution artifact."""
    if type(execution) is not AgentExecutionResult:
        raise TypeError("execution must be AgentExecutionResult")
    if type(identity_seed) is not DecisionFlowIdentitySeed:
        raise TypeError("identity_seed must be DecisionFlowIdentitySeed")
    if type(flow_scope) is not DecisionFlowScope:
        raise TypeError("flow_scope must be DecisionFlowScope")
    resolved_kind = (
        artifact_kind if artifact_kind is not None else _AGENT_EXECUTION_ARTIFACT_KIND
    )
    return DecisionFlowRequest(
        identity_seed=identity_seed,
        artifact_kind=resolved_kind,
        payload=execution,
        flow_scope=flow_scope,
    )


async def evaluate_agent_execution_flow(
    gate: DecisionFlowGate[AgentExecutionResult],
    request: DecisionFlowRequest[AgentExecutionResult],
) -> DecisionFlowResult[AgentExecutionResult]:
    """Evaluate one agent-execution decision request through a configured gate."""
    if type(request) is not DecisionFlowRequest:
        raise TypeError("request must be DecisionFlowRequest")
    return await gate.evaluate(request)


def decision_flow_result_to_validation_result(
    result: DecisionFlowResult[AgentExecutionResult],
) -> ValidationResult:
    """Map canonical decision-flow host action to Nexus validation semantics."""
    if type(result) is not DecisionFlowResult:
        raise TypeError("result must be DecisionFlowResult")
    if result.host_action is DecisionFlowHostAction.CONTINUE:
        return ValidationResult(valid=True)
    errors = [result.authority_reason or "decision_flow_blocked"]
    return ValidationResult(valid=False, errors=errors)
