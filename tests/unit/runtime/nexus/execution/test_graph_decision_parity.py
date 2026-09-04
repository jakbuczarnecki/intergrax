# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.decision_revision import decision_revision_policy
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.runtime.decision_flow import (
    CanonicalDecisionFlowGate,
    DecisionFlowGateCapabilities,
    DecisionFlowScope,
)
from intergrax.runtime.decision_flow_host import (
    build_agent_execution_flow_request,
    build_agent_execution_verification_pipeline,
    decision_flow_result_to_validation_result,
    evaluate_agent_execution_flow,
    agent_execution_decision_context,
    agent_execution_identity_seed,
)
from intergrax.runtime.execution.active_decision_lifecycle_host import (
    bind_active_decision_lifecycle_host,
    reset_active_decision_lifecycle_host,
)
from intergrax.runtime.execution.decision_lifecycle_host import CanonicalDecisionLifecycleHost

pytestmark = pytest.mark.unit


def _structural_contract() -> AgentContract:
    return AgentContract(
        id="agent-a",
        name="agent-a",
        description="parity",
        validation_rules=["non_empty_summary"],
    )


def _build_gate(*, contract: AgentContract) -> CanonicalDecisionFlowGate[AgentExecutionResult]:
    return CanonicalDecisionFlowGate(
        capabilities=DecisionFlowGateCapabilities(
            verification_pipeline=build_agent_execution_verification_pipeline(
                contract=contract,
            ),
            revision_policy=decision_revision_policy(max_revisions=0),
            scopes=frozenset({DecisionFlowScope.GRAPH_FINAL}),
        ),
    )


def _execution(*, summary: str) -> AgentExecutionResult:
    return AgentExecutionResult(
        agent_id="agent-a",
        run_id="run-1",
        status=AgentExecutionStatus.COMPLETED,
        summary=summary,
    )


@pytest.fixture
def lifecycle_binding():
    token = bind_active_decision_lifecycle_host(CanonicalDecisionLifecycleHost())
    yield
    reset_active_decision_lifecycle_host(token)


@pytest.fixture
def execution_identity_binding():
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    yield task_id, run_id, attempt_id
    reset_active_execution_identity(token)


@pytest.mark.asyncio
async def test_graph_decision_validation_is_stable(
    lifecycle_binding,
    execution_identity_binding,
) -> None:
    contract = _structural_contract()
    gate = _build_gate(contract=contract)
    task_id, run_id, attempt_id = execution_identity_binding
    execution = _execution(summary="valid summary")
    decision_context = agent_execution_decision_context(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id="tenant-1",
    )
    identity_seed = agent_execution_identity_seed(
        context=decision_context,
        namespace="graph.final",
        subject="graph-1",
    )
    flow_request = build_agent_execution_flow_request(
        execution=execution,
        identity_seed=identity_seed,
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
    )
    first = await evaluate_agent_execution_flow(gate, flow_request)
    second = await evaluate_agent_execution_flow(gate, flow_request)
    assert decision_flow_result_to_validation_result(first) == decision_flow_result_to_validation_result(second)


@pytest.mark.asyncio
async def test_graph_decision_structural_failure(
    lifecycle_binding,
    execution_identity_binding,
) -> None:
    contract = _structural_contract()
    gate = _build_gate(contract=contract)
    task_id, run_id, attempt_id = execution_identity_binding
    execution = _execution(summary="")
    decision_context = agent_execution_decision_context(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id="tenant-1",
    )
    identity_seed = agent_execution_identity_seed(
        context=decision_context,
        namespace="graph.final",
        subject="graph-1",
    )
    flow_request = build_agent_execution_flow_request(
        execution=execution,
        identity_seed=identity_seed,
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
    )
    flow_result = await evaluate_agent_execution_flow(gate, flow_request)
    validation = decision_flow_result_to_validation_result(flow_result)
    assert validation.valid is False
