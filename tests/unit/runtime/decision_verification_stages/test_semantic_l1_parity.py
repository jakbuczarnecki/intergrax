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
from intergrax.contracts.semantic_verification import (
    ResolvedSemanticRubric,
    SemanticRubricRef,
    resolved_semantic_rubric,
    semantic_rubric_ref,
    semantic_verification_independence_config,
    VerifierIndependenceMode,
)
from intergrax.runtime.critic.contracts import CriticRequest, CriticScope, RubricSpec
from intergrax.runtime.critic.l1_gateway import L1Gateway
from intergrax.runtime.decision_verification_stages.semantic import SemanticVerificationStage
from intergrax.runtime.execution.inference_profile import InferenceProfileId
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer
from intergrax.tools.providers.eval.contracts import EvalJudgeInput, EvalJudgeOutput
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)


@dataclass(frozen=True, slots=True)
class ParityPayload:
    text: str


@dataclass(frozen=True, slots=True)
class ParityContentExtractor:
    def extract(self, candidate: CandidateDecision[ParityPayload]) -> str:
        return candidate.artifact.content.text


@dataclass(frozen=True, slots=True)
class SharedEvalJudge:
    passed: bool

    def is_available(self) -> bool:
        return True

    def judge(self, params: EvalJudgeInput) -> EvalJudgeOutput:
        return EvalJudgeOutput(
            rubric_id=params.rubric_id,
            score=1.0 if self.passed else 0.2,
            passed=self.passed,
            reasons=() if self.passed else ("below threshold",),
        )


@dataclass(frozen=True, slots=True)
class SharedToolClient:
    judge_impl: SharedEvalJudge

    def judge(self, params: EvalJudgeInput) -> EvalJudgeOutput:
        return self.judge_impl.judge(params)

    def trajectory(self, params: object) -> object:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ParityRubricResolver:
    rubric: ResolvedSemanticRubric

    def is_available(self) -> bool:
        return True

    def resolve(self, ref: SemanticRubricRef) -> ResolvedSemanticRubric:
        return self.rubric


def _candidate(text: str) -> CandidateDecision[ParityPayload]:
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="parity", subject="subject-1"),
        tenant_id="tenant-a",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )
    artifact = DecisionArtifact(
        kind=validate_decision_artifact_kind("parity_payload"),
        content=ParityPayload(text=text),
    )
    lineage = DecisionVersionLineage(
        current=decision_lineage_ref(identity.version),
    )
    return CandidateDecision(identity=identity, artifact=artifact, lineage=lineage)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_semantic_stage_matches_l1_gateway_pass_fail_semantics() -> None:
    ref = semantic_rubric_ref(rubric_id="case.a", version=1)
    rubric = resolved_semantic_rubric(
        ref=ref,
        criteria=("Criterion one",),
        min_score=0.75,
        provenance_ref="prompt_registry:case.a@1",
    )
    for passed in (True, False):
        judge = SharedEvalJudge(passed=passed)
        stage = SemanticVerificationStage(
            rubric_ref=rubric.ref,
            rubric_resolver=ParityRubricResolver(rubric=rubric),
            content_provider=ParityContentExtractor(),
            judge=judge,
            independence=semantic_verification_independence_config(
                mode=VerifierIndependenceMode.INDEPENDENT,
                producer_profile_id=InferenceProfileId("producer"),
                verifier_profile_id=InferenceProfileId("verifier"),
            ),
        )
        candidate = _candidate("bounded output")
        stage_record = await stage.verify(candidate)
        gateway = L1Gateway(tool_client=SharedToolClient(judge_impl=judge))
        legacy = gateway.verify_semantic(
            CriticRequest(
                scope=CriticScope.GRAPH_FINAL,
                run_id=str(candidate.identity.execution.run_id),
                agent_id="agent-1",
                answer=RuntimeAnswer(answer=candidate.artifact.content.text),
                rubric=RubricSpec(
                    rubric_id=str(rubric.ref.rubric_id),
                    criteria=list(rubric.criteria),
                    reference_context=rubric.reference_context,
                    min_score=rubric.min_score,
                ),
            ),
        )
        assert stage_record.outcome.value == ("passed" if legacy.passed else "challenged")
