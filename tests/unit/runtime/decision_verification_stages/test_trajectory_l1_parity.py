# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.contracts.decision_record import (
    CandidateDecision,
    DecisionArtifact,
    DecisionVersionLineage,
    decision_lineage_ref,
    validate_decision_artifact_kind,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.trajectory_verification import TrajectoryAgentId
from intergrax.runtime.critic.contracts import CriticRequest, CriticScope
from intergrax.runtime.critic.l1_gateway import L1Gateway
from intergrax.runtime.decision_verification_stages.trajectory import TrajectoryVerificationStage
from intergrax.tools.providers.eval.contracts import (
    EvalJudgeInput,
    EvalJudgeOutput,
    EvalTrajectoryInput,
    EvalTrajectoryOutput,
)


@dataclass(frozen=True, slots=True)
class ParityPayload:
    text: str


@dataclass(frozen=True, slots=True)
class ParityAgentProvider:
    agent_id: TrajectoryAgentId

    def resolve(self, candidate: CandidateDecision[ParityPayload]) -> TrajectoryAgentId:
        return self.agent_id


@dataclass(frozen=True, slots=True)
class SharedTrajectoryEvaluator:
    passed: bool

    def is_available(self) -> bool:
        return True

    def evaluate(self, params: EvalTrajectoryInput) -> EvalTrajectoryOutput:
        return EvalTrajectoryOutput(
            run_id=params.run_id,
            score=1.0 if self.passed else 0.2,
            passed=self.passed,
            reasons=[] if self.passed else ["below threshold"],
        )


@dataclass(frozen=True, slots=True)
class SharedToolClient:
    trajectory_impl: SharedTrajectoryEvaluator

    def judge(self, params: EvalJudgeInput) -> EvalJudgeOutput:
        raise NotImplementedError

    def trajectory(self, params: EvalTrajectoryInput) -> EvalTrajectoryOutput:
        return self.trajectory_impl.evaluate(params)


def _candidate(tenant_id: str = "tenant-a") -> CandidateDecision[ParityPayload]:
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="parity", subject="subject-1"),
        tenant_id=tenant_id,
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )
    artifact = DecisionArtifact(
        kind=validate_decision_artifact_kind("parity_payload"),
        content=ParityPayload(text="ok"),
    )
    lineage = DecisionVersionLineage(
        current=decision_lineage_ref(identity.version),
    )
    return CandidateDecision(identity=identity, artifact=artifact, lineage=lineage)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_trajectory_stage_matches_l1_gateway_pass_fail_semantics() -> None:
    for passed in (True, False):
        evaluator = SharedTrajectoryEvaluator(passed=passed)
        candidate = _candidate(tenant_id="tenant-xyz")
        stage = TrajectoryVerificationStage(
            evaluator=evaluator,
            agent_id_provider=ParityAgentProvider(agent_id=TrajectoryAgentId("agent-bound")),
        )
        stage_record = await stage.verify(candidate)
        gateway = L1Gateway(tool_client=SharedToolClient(trajectory_impl=evaluator))
        legacy = gateway.verify_trajectory(
            CriticRequest(
                scope=CriticScope.GRAPH_FINAL,
                run_id=str(candidate.identity.execution.run_id),
                agent_id="agent-bound",
                context={
                    "tenant_id": candidate.identity.tenant_id,
                    "trajectory_min_score": 0.75,
                },
            ),
        )
        assert stage_record.outcome.value == ("passed" if legacy.passed else "challenged")
