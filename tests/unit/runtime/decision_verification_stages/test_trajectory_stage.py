# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    candidate_decision_ref,
    decision_lineage_ref,
    validate_decision_artifact_kind,
)
from intergrax.contracts.decision_verification import (
    VerificationDisposition,
    VerificationStageOutcome,
)
from intergrax.contracts.decision_verification_stage import (
    VerificationStageExecutionClass,
    VerificationStageRegistration,
    VerificationStageUnavailableError,
    verification_stage_registry,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.trajectory_verification import (
    TrajectoryAgentId,
    TrajectoryVerificationStageConfig,
    trajectory_verification_stage_config,
)
from intergrax.runtime.decision_verification import VerificationPipeline
from intergrax.runtime.decision_verification_stages.structural import (
    STRUCTURAL_VERIFICATION_STAGE_KIND,
    NonEmptyTextStructuralValidator,
    StructuralVerificationStage,
)
from intergrax.runtime.decision_verification_stages.trajectory import (
    TRAJECTORY_VERIFICATION_STAGE_KIND,
    TrajectoryVerificationStage,
)
from intergrax.tools.providers.eval.contracts import EvalTrajectoryInput, EvalTrajectoryOutput

_MODULE_PATHS = (
    Path("intergrax/contracts/trajectory_verification.py"),
    Path("intergrax/runtime/decision_verification_stages/trajectory.py"),
)
_FORBIDDEN_FRAGMENTS = (
    "runtime.critic",
    "runtime.nexus",
    "L1Gateway",
    "CriticEvalToolClient",
    "CriticOrchestrator",
    "Any",
    "cast(",
    "type: ignore",
    "pyright: ignore",
    "getattr",
    "setattr",
    "hasattr",
    "inspect",
    "exec(",
    "eval(",
    "object.__setattr__",
    "dict[str, Any]",
)


@dataclass(frozen=True, slots=True)
class TrajectoryPayload:
    text: str


@dataclass(frozen=True, slots=True)
class FixedAgentIdProvider:
    agent_id: TrajectoryAgentId

    def resolve(self, candidate: CandidateDecision[TrajectoryPayload]) -> TrajectoryAgentId:
        return self.agent_id


@dataclass(frozen=True, slots=True)
class RecordingTrajectoryEvaluator:
    available: bool = True
    passed: bool = True
    calls: list[EvalTrajectoryInput] | None = None

    def is_available(self) -> bool:
        return self.available

    def evaluate(self, params: EvalTrajectoryInput) -> EvalTrajectoryOutput:
        if self.calls is not None:
            self.calls.append(params)
        return EvalTrajectoryOutput(
            run_id=params.run_id,
            score=1.0 if self.passed else 0.2,
            passed=self.passed,
            reasons=[] if self.passed else ["below threshold"],
        )


def _execution_lineage() -> DecisionExecutionLineage:
    return DecisionExecutionLineage(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _candidate(tenant_id: str = "tenant-a") -> CandidateDecision[TrajectoryPayload]:
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="trajectory", subject="subject-1"),
        tenant_id=tenant_id,
        execution=_execution_lineage(),
    )
    artifact = DecisionArtifact(
        kind=validate_decision_artifact_kind("trajectory_payload"),
        content=TrajectoryPayload(text="ok"),
    )
    lineage = DecisionVersionLineage(
        current=decision_lineage_ref(identity.version),
    )
    return CandidateDecision(
        identity=identity,
        artifact=artifact,
        lineage=lineage,
    )


def _stage(
    *,
    evaluator: RecordingTrajectoryEvaluator,
    agent_id: str = "agent-explicit",
    config: TrajectoryVerificationStageConfig | None = None,
) -> TrajectoryVerificationStage[TrajectoryPayload]:
    return TrajectoryVerificationStage(
        evaluator=evaluator,
        agent_id_provider=FixedAgentIdProvider(agent_id=TrajectoryAgentId(agent_id)),
        config=config or trajectory_verification_stage_config(),
    )


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_evaluator_pass_produces_passed() -> None:
    stage = _stage(evaluator=RecordingTrajectoryEvaluator(passed=True))
    record = await stage.verify(_candidate())
    assert record.outcome is VerificationStageOutcome.PASSED


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_evaluator_fail_produces_challenged() -> None:
    stage = _stage(evaluator=RecordingTrajectoryEvaluator(passed=False))
    record = await stage.verify(_candidate())
    assert record.outcome is VerificationStageOutcome.CHALLENGED
    assert record.challenge is not None
    assert record.challenge.finding.code == "verification.trajectory.below_requirement"


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_candidate_run_identity_passed_exactly() -> None:
    candidate = _candidate()
    evaluator = RecordingTrajectoryEvaluator(calls=[])
    stage = _stage(evaluator=evaluator)
    _ = await stage.verify(candidate)
    assert evaluator.calls is not None
    assert len(evaluator.calls) == 1
    assert evaluator.calls[0].run_id == str(candidate.identity.execution.run_id)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_candidate_tenant_passed_exactly() -> None:
    candidate = _candidate(tenant_id="tenant-xyz")
    evaluator = RecordingTrajectoryEvaluator(calls=[])
    stage = _stage(evaluator=evaluator)
    _ = await stage.verify(candidate)
    assert evaluator.calls is not None
    assert evaluator.calls[0].tenant_id == "tenant-xyz"


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_agent_identity_explicit() -> None:
    evaluator = RecordingTrajectoryEvaluator(calls=[])
    stage = _stage(evaluator=evaluator, agent_id="agent-bound")
    _ = await stage.verify(_candidate())
    assert evaluator.calls is not None
    assert evaluator.calls[0].agent_id == "agent-bound"


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_missing_required_agent_identity_challenged() -> None:
    @dataclass(frozen=True, slots=True)
    class MissingAgentProvider:
        def resolve(self, candidate: CandidateDecision[TrajectoryPayload]) -> TrajectoryAgentId:
            return TrajectoryAgentId("")

    evaluator = RecordingTrajectoryEvaluator(calls=[])
    stage = TrajectoryVerificationStage(
        evaluator=evaluator,
        agent_id_provider=MissingAgentProvider(),
    )
    record = await stage.verify(_candidate())
    assert record.outcome is VerificationStageOutcome.CHALLENGED
    assert record.challenge is not None
    assert record.challenge.finding.code == "verification.trajectory.agent_id_missing"
    assert evaluator.calls == []


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_evaluator_unavailable_raises() -> None:
    stage = _stage(evaluator=RecordingTrajectoryEvaluator(available=False))
    with pytest.raises(VerificationStageUnavailableError):
        await stage.verify(_candidate())


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_deterministic_challenge_prevents_trajectory_invocation() -> None:
    evaluator = RecordingTrajectoryEvaluator(calls=[])
    trajectory = _stage(evaluator=evaluator)

    @dataclass(frozen=True, slots=True)
    class TextExtractor:
        def extract(self, content: TrajectoryPayload) -> str:
            return content.text

    structural = StructuralVerificationStage(
        validators=(
            NonEmptyTextStructuralValidator(
                extractor=TextExtractor(),
                field_label="text",
            ),
        ),
    )
    candidate = _candidate()
    candidate = CandidateDecision(
        identity=candidate.identity,
        artifact=DecisionArtifact(
            kind=candidate.artifact.kind,
            content=TrajectoryPayload(text="   "),
        ),
        lineage=candidate.lineage,
    )
    registrations = (
        VerificationStageRegistration(
            kind=STRUCTURAL_VERIFICATION_STAGE_KIND,
            stage=structural,
            required=True,
        ),
        VerificationStageRegistration(
            kind=TRAJECTORY_VERIFICATION_STAGE_KIND,
            stage=trajectory,
            required=True,
        ),
    )
    pipeline = VerificationPipeline(registry=verification_stage_registry(registrations))
    _ = await pipeline.verify(candidate)
    assert evaluator.calls == []


@pytest.mark.unit
@pytest.mark.gate
def test_execution_class_probabilistic() -> None:
    stage = _stage(evaluator=RecordingTrajectoryEvaluator())
    assert stage.execution_class is VerificationStageExecutionClass.PROBABILISTIC


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_record_observation_false() -> None:
    evaluator = RecordingTrajectoryEvaluator(calls=[])
    stage = _stage(evaluator=evaluator)
    _ = await stage.verify(_candidate())
    assert evaluator.calls is not None
    assert evaluator.calls[0].record_observation is False


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_threshold_from_typed_config() -> None:
    evaluator = RecordingTrajectoryEvaluator(calls=[])
    stage = _stage(
        evaluator=evaluator,
        config=trajectory_verification_stage_config(min_score=0.9),
    )
    _ = await stage.verify(_candidate())
    assert evaluator.calls is not None
    assert evaluator.calls[0].min_score == 0.9


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_exact_proposal_identity() -> None:
    candidate = _candidate()
    stage = _stage(evaluator=RecordingTrajectoryEvaluator())
    record = await stage.verify(candidate)
    assert record.proposal_ref == candidate_decision_ref(candidate)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_required_unavailable_integrates_with_pipeline() -> None:
    stage = _stage(evaluator=RecordingTrajectoryEvaluator(available=False))
    registration = VerificationStageRegistration(
        kind=TRAJECTORY_VERIFICATION_STAGE_KIND,
        stage=stage,
        required=True,
    )
    pipeline = VerificationPipeline(registry=verification_stage_registry((registration,)))
    result = await pipeline.verify(_candidate())
    assert result.disposition is VerificationDisposition.CHALLENGED


@pytest.mark.unit
@pytest.mark.gate
def test_no_critic_dependency() -> None:
    for path in _MODULE_PATHS:
        source = path.read_text(encoding="utf-8")
        assert "runtime.critic" not in source


@pytest.mark.unit
@pytest.mark.gate
def test_forbidden_audit_trajectory_modules() -> None:
    for path in _MODULE_PATHS:
        source = path.read_text(encoding="utf-8")
        for fragment in _FORBIDDEN_FRAGMENTS:
            assert fragment not in source
