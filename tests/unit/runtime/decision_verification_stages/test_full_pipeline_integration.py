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
from intergrax.contracts.decision_verification import (
    VerificationDisposition,
    VerificationStageOutcome,
    validate_verification_finding_code,
    validate_verification_requirement_code,
)
from intergrax.contracts.decision_verification_stage import (
    VerificationStageExecutionClass,
    VerificationStageRegistration,
    verification_stage_registry,
)
from intergrax.contracts.domain_verification import (
    DomainVerificationOutcome,
    DomainVerifierId,
    DomainVerificationIndependenceConfig,
    domain_verification_passed,
)
from intergrax.contracts.evidence_claims import (
    EvidenceBackedClaim,
    EvidenceClaimSet,
    mint_evidence_claim_id,
    validate_claim_kind,
    validate_evidence_reference_id,
)
from intergrax.contracts.evidence_verification import EvidenceClaimsProvider, EvidenceReferenceResolver
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.semantic_verification import (
    ResolvedSemanticRubric,
    SemanticRubricNotFoundError,
    SemanticRubricRef,
    VerifierIndependenceMode,
    resolved_semantic_rubric,
    semantic_rubric_ref,
    semantic_verification_independence_config,
)
from intergrax.contracts.trajectory_verification import TrajectoryAgentId
from intergrax.runtime.decision_verification import VerificationPipeline
from intergrax.runtime.decision_verification_stages.domain import (
    DOMAIN_VERIFICATION_STAGE_KIND,
    IndependentDomainVerificationStage,
)
from intergrax.runtime.decision_verification_stages.evidence import (
    EVIDENCE_VERIFICATION_STAGE_KIND,
    EvidenceVerificationStage,
)
from intergrax.runtime.decision_verification_stages.semantic import (
    SEMANTIC_VERIFICATION_STAGE_KIND,
    SemanticVerificationStage,
)
from intergrax.runtime.decision_verification_stages.structural import (
    STRUCTURAL_VERIFICATION_STAGE_KIND,
    NonEmptyTextStructuralValidator,
    StructuralVerificationStage,
)
from intergrax.runtime.decision_verification_stages.trajectory import (
    TRAJECTORY_VERIFICATION_STAGE_KIND,
    TrajectoryVerificationStage,
)
from intergrax.runtime.execution.inference_profile import InferenceProfileId
from intergrax.tools.providers.eval.contracts import EvalJudgeInput, EvalJudgeOutput, EvalTrajectoryInput, EvalTrajectoryOutput


@dataclass(frozen=True, slots=True)
class FullPipelinePayload:
    text: str
    claim_set: EvidenceClaimSet | None


@dataclass(frozen=True, slots=True)
class FullTextExtractor:
    def extract(self, content: FullPipelinePayload) -> str:
        return content.text


@dataclass(frozen=True, slots=True)
class FullSemanticExtractor:
    def extract(self, candidate: CandidateDecision[FullPipelinePayload]) -> str:
        return candidate.artifact.content.text


@dataclass(frozen=True, slots=True)
class FullEvidenceExtractor:
    def extract(self, candidate: CandidateDecision[FullPipelinePayload]) -> EvidenceClaimSet | None:
        return candidate.artifact.content.claim_set


@dataclass(frozen=True, slots=True)
class FullEvidenceResolver:
    known_ids: frozenset[str]

    def is_available(self) -> bool:
        return True

    def evidence_exists(self, evidence_id: object) -> bool:
        return str(evidence_id) in self.known_ids


@dataclass(frozen=True, slots=True)
class FullRubricResolver:
    rubric: ResolvedSemanticRubric

    def is_available(self) -> bool:
        return True

    def resolve(self, ref: SemanticRubricRef) -> ResolvedSemanticRubric:
        if ref.rubric_id != self.rubric.ref.rubric_id or ref.version != self.rubric.ref.version:
            raise SemanticRubricNotFoundError("missing")
        return self.rubric


@dataclass(frozen=True, slots=True)
class RecordingJudge:
    calls: list[EvalJudgeInput]

    def is_available(self) -> bool:
        return True

    def judge(self, params: EvalJudgeInput) -> EvalJudgeOutput:
        self.calls.append(params)
        return EvalJudgeOutput(
            rubric_id=params.rubric_id,
            score=1.0,
            passed=True,
            reasons=[],
        )


@dataclass(frozen=True, slots=True)
class RecordingTrajectoryEvaluator:
    calls: list[EvalTrajectoryInput]

    def is_available(self) -> bool:
        return True

    def evaluate(self, params: EvalTrajectoryInput) -> EvalTrajectoryOutput:
        self.calls.append(params)
        return EvalTrajectoryOutput(
            run_id=params.run_id,
            score=1.0,
            passed=True,
            reasons=[],
        )


@dataclass(frozen=True, slots=True)
class FullAgentProvider:
    agent_id: TrajectoryAgentId

    def resolve(self, candidate: CandidateDecision[FullPipelinePayload]) -> TrajectoryAgentId:
        return self.agent_id


@dataclass(frozen=True, slots=True)
class FullDomainVerifier:
    verifier_id_value: DomainVerifierId
    calls: list[CandidateDecision[FullPipelinePayload]]

    @property
    def verifier_id(self) -> DomainVerifierId:
        return self.verifier_id_value

    def is_available(self) -> bool:
        return True

    def verify(self, candidate: CandidateDecision[FullPipelinePayload]) -> DomainVerificationOutcome:
        self.calls.append(candidate)
        return domain_verification_passed()


def _candidate() -> CandidateDecision[FullPipelinePayload]:
    evidence_id = validate_evidence_reference_id("evidence.ref.1")
    claim = EvidenceBackedClaim(
        claim_id=mint_evidence_claim_id(),
        statement="Bounded claim.",
        claim_kind=validate_claim_kind("generic.claim"),
        supporting_evidence_ids=(evidence_id,),
    )
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="full", subject="subject-1"),
        tenant_id="tenant-a",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )
    artifact = DecisionArtifact(
        kind=validate_decision_artifact_kind("full_payload"),
        content=FullPipelinePayload(
            text="bounded answer",
            claim_set=EvidenceClaimSet(claims=(claim,)),
        ),
    )
    lineage = DecisionVersionLineage(
        current=decision_lineage_ref(identity.version),
    )
    return CandidateDecision(identity=identity, artifact=artifact, lineage=lineage)


def _full_pipeline(
    *,
    judge_calls: list[EvalJudgeInput],
    trajectory_calls: list[EvalTrajectoryInput],
    domain_calls: list[CandidateDecision[FullPipelinePayload]],
    semantic_producer: str = "profile-a",
    semantic_verifier: str = "profile-b",
    domain_producer: str = "profile-a",
    domain_verifier: str = "profile-c",
) -> VerificationPipeline[FullPipelinePayload]:
    rubric = resolved_semantic_rubric(
        ref=semantic_rubric_ref(rubric_id="quality.summary", version=1),
        criteria=("Criterion",),
        min_score=0.75,
        provenance_ref="prompt_registry:quality.summary@1",
    )
    evidence_id = validate_evidence_reference_id("evidence.ref.1")
    structural = StructuralVerificationStage(
        validators=(
            NonEmptyTextStructuralValidator(
                extractor=FullTextExtractor(),
                field_label="text",
            ),
        ),
    )
    evidence = EvidenceVerificationStage(
        claims_provider=FullEvidenceExtractor(),
        resolver=FullEvidenceResolver(known_ids=frozenset({str(evidence_id)})),
    )
    semantic = SemanticVerificationStage(
        rubric_ref=rubric.ref,
        rubric_resolver=FullRubricResolver(rubric=rubric),
        content_provider=FullSemanticExtractor(),
        judge=RecordingJudge(calls=judge_calls),
        independence=semantic_verification_independence_config(
            mode=VerifierIndependenceMode.INDEPENDENT,
            producer_profile_id=semantic_producer,
            verifier_profile_id=semantic_verifier,
        ),
    )
    trajectory = TrajectoryVerificationStage(
        evaluator=RecordingTrajectoryEvaluator(calls=trajectory_calls),
        agent_id_provider=FullAgentProvider(agent_id=TrajectoryAgentId("agent-bound")),
    )
    domain_verifier_impl = FullDomainVerifier(
        verifier_id_value=DomainVerifierId("domain.legal"),
        calls=domain_calls,
    )
    domain = IndependentDomainVerificationStage(
        verifier=domain_verifier_impl,
        execution_class=VerificationStageExecutionClass.PROBABILISTIC,
        independence=DomainVerificationIndependenceConfig(
            mode=VerifierIndependenceMode.INDEPENDENT,
            producer_profile_id=InferenceProfileId(domain_producer),
            verifier_profile_id=InferenceProfileId(domain_verifier),
        ),
    )
    registrations = (
        VerificationStageRegistration(
            kind=STRUCTURAL_VERIFICATION_STAGE_KIND,
            stage=structural,
            required=True,
        ),
        VerificationStageRegistration(
            kind=EVIDENCE_VERIFICATION_STAGE_KIND,
            stage=evidence,
            required=True,
        ),
        VerificationStageRegistration(
            kind=SEMANTIC_VERIFICATION_STAGE_KIND,
            stage=semantic,
            required=True,
        ),
        VerificationStageRegistration(
            kind=TRAJECTORY_VERIFICATION_STAGE_KIND,
            stage=trajectory,
            required=True,
        ),
        VerificationStageRegistration(
            kind=DOMAIN_VERIFICATION_STAGE_KIND,
            stage=domain,
            required=True,
        ),
    )
    return VerificationPipeline(registry=verification_stage_registry(registrations))


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_full_pipeline_executes_deterministic_before_probabilistic() -> None:
    judge_calls: list[EvalJudgeInput] = []
    trajectory_calls: list[EvalTrajectoryInput] = []
    domain_calls: list[CandidateDecision[FullPipelinePayload]] = []
    pipeline = _full_pipeline(
        judge_calls=judge_calls,
        trajectory_calls=trajectory_calls,
        domain_calls=domain_calls,
    )
    result = await pipeline.verify(_candidate())
    assert result.disposition is VerificationDisposition.PASSED
    stage_kinds = [record.stage for record in result.stage_records]
    deterministic_kinds = {
        STRUCTURAL_VERIFICATION_STAGE_KIND,
        EVIDENCE_VERIFICATION_STAGE_KIND,
    }
    probabilistic_kinds = {
        SEMANTIC_VERIFICATION_STAGE_KIND,
        TRAJECTORY_VERIFICATION_STAGE_KIND,
        DOMAIN_VERIFICATION_STAGE_KIND,
    }
    assert deterministic_kinds.issubset(set(stage_kinds))
    assert probabilistic_kinds.issubset(set(stage_kinds))
    last_deterministic_index = max(
        stage_kinds.index(kind) for kind in deterministic_kinds
    )
    first_probabilistic_index = min(
        stage_kinds.index(kind) for kind in probabilistic_kinds
    )
    assert last_deterministic_index < first_probabilistic_index
    assert judge_calls
    assert trajectory_calls
    assert domain_calls


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_deterministic_challenge_blocks_all_probabilistic_verifiers() -> None:
    judge_calls: list[EvalJudgeInput] = []
    trajectory_calls: list[EvalTrajectoryInput] = []
    domain_calls: list[CandidateDecision[FullPipelinePayload]] = []
    pipeline = _full_pipeline(
        judge_calls=judge_calls,
        trajectory_calls=trajectory_calls,
        domain_calls=domain_calls,
    )
    candidate = _candidate()
    candidate = CandidateDecision(
        identity=candidate.identity,
        artifact=DecisionArtifact(
            kind=candidate.artifact.kind,
            content=FullPipelinePayload(text="   ", claim_set=None),
        ),
        lineage=candidate.lineage,
    )
    result = await pipeline.verify(candidate)
    assert result.disposition is VerificationDisposition.CHALLENGED
    assert judge_calls == []
    assert trajectory_calls == []
    assert domain_calls == []


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_semantic_and_domain_independence_profiles_observable() -> None:
    judge_calls: list[EvalJudgeInput] = []
    trajectory_calls: list[EvalTrajectoryInput] = []
    domain_calls: list[CandidateDecision[FullPipelinePayload]] = []
    pipeline = _full_pipeline(
        judge_calls=judge_calls,
        trajectory_calls=trajectory_calls,
        domain_calls=domain_calls,
        semantic_producer="profile-a",
        semantic_verifier="profile-b",
        domain_producer="profile-a",
        domain_verifier="profile-c",
    )
    result = await pipeline.verify(_candidate())
    assert result.disposition is VerificationDisposition.PASSED
    semantic_registration = next(
        stage for stage in pipeline.registry.registrations
        if stage.kind == SEMANTIC_VERIFICATION_STAGE_KIND
    )
    domain_registration = next(
        stage for stage in pipeline.registry.registrations
        if stage.kind == DOMAIN_VERIFICATION_STAGE_KIND
    )
    semantic_stage = semantic_registration.stage
    domain_stage = domain_registration.stage
    assert isinstance(semantic_stage, SemanticVerificationStage)
    assert isinstance(domain_stage, IndependentDomainVerificationStage)
    semantic_independence = semantic_stage.independence
    domain_independence = domain_stage.independence
    assert semantic_independence is not None
    assert domain_independence is not None
    assert semantic_independence.producer_profile_id == InferenceProfileId("profile-a")
    assert semantic_independence.verifier_profile_id == InferenceProfileId("profile-b")
    assert domain_independence.verifier_profile_id == InferenceProfileId("profile-c")


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_required_independence_same_profile_challenges_semantic_before_judge() -> None:
    judge_calls: list[EvalJudgeInput] = []
    trajectory_calls: list[EvalTrajectoryInput] = []
    domain_calls: list[CandidateDecision[FullPipelinePayload]] = []
    pipeline = _full_pipeline(
        judge_calls=judge_calls,
        trajectory_calls=trajectory_calls,
        domain_calls=domain_calls,
        semantic_producer="profile-a",
        semantic_verifier="profile-a",
    )
    result = await pipeline.verify(_candidate())
    assert result.disposition is VerificationDisposition.CHALLENGED
    semantic_record = next(
        record for record in result.stage_records
        if record.stage == SEMANTIC_VERIFICATION_STAGE_KIND
    )
    assert semantic_record.outcome is VerificationStageOutcome.CHALLENGED
    assert semantic_record.challenge is not None
    assert semantic_record.challenge.finding.code == "verification.semantic.profile_not_independent"
    assert judge_calls == []


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_semantic_rubric_ref_mismatch_challenges_pipeline_without_judge() -> None:
    judge_calls: list[EvalJudgeInput] = []
    trajectory_calls: list[EvalTrajectoryInput] = []
    domain_calls: list[CandidateDecision[FullPipelinePayload]] = []
    configured = semantic_rubric_ref(rubric_id="quality.summary", version=2)
    wrong_version = resolved_semantic_rubric(
        ref=semantic_rubric_ref(rubric_id="quality.summary", version=1),
        criteria=("Criterion",),
        min_score=0.75,
        provenance_ref="prompt_registry:quality.summary@1",
    )

    @dataclass(frozen=True, slots=True)
    class VersionDriftResolver:
        rubric: ResolvedSemanticRubric

        def is_available(self) -> bool:
            return True

        def resolve(self, ref: SemanticRubricRef) -> ResolvedSemanticRubric:
            return self.rubric

    evidence_id = validate_evidence_reference_id("evidence.ref.1")
    structural = StructuralVerificationStage(
        validators=(
            NonEmptyTextStructuralValidator(
                extractor=FullTextExtractor(),
                field_label="text",
            ),
        ),
    )
    evidence = EvidenceVerificationStage(
        claims_provider=FullEvidenceExtractor(),
        resolver=FullEvidenceResolver(known_ids=frozenset({str(evidence_id)})),
    )
    semantic = SemanticVerificationStage(
        rubric_ref=configured,
        rubric_resolver=VersionDriftResolver(rubric=wrong_version),
        content_provider=FullSemanticExtractor(),
        judge=RecordingJudge(calls=judge_calls),
        independence=semantic_verification_independence_config(
            mode=VerifierIndependenceMode.INDEPENDENT,
            producer_profile_id="profile-a",
            verifier_profile_id="profile-b",
        ),
    )
    trajectory = TrajectoryVerificationStage(
        evaluator=RecordingTrajectoryEvaluator(calls=trajectory_calls),
        agent_id_provider=FullAgentProvider(agent_id=TrajectoryAgentId("agent-bound")),
    )
    domain_verifier_impl = FullDomainVerifier(
        verifier_id_value=DomainVerifierId("domain.legal"),
        calls=domain_calls,
    )
    domain = IndependentDomainVerificationStage(
        verifier=domain_verifier_impl,
        execution_class=VerificationStageExecutionClass.PROBABILISTIC,
        independence=DomainVerificationIndependenceConfig(
            mode=VerifierIndependenceMode.INDEPENDENT,
            producer_profile_id=InferenceProfileId("profile-a"),
            verifier_profile_id=InferenceProfileId("profile-c"),
        ),
    )
    registrations = (
        VerificationStageRegistration(
            kind=STRUCTURAL_VERIFICATION_STAGE_KIND,
            stage=structural,
            required=True,
        ),
        VerificationStageRegistration(
            kind=EVIDENCE_VERIFICATION_STAGE_KIND,
            stage=evidence,
            required=True,
        ),
        VerificationStageRegistration(
            kind=SEMANTIC_VERIFICATION_STAGE_KIND,
            stage=semantic,
            required=True,
        ),
        VerificationStageRegistration(
            kind=TRAJECTORY_VERIFICATION_STAGE_KIND,
            stage=trajectory,
            required=True,
        ),
        VerificationStageRegistration(
            kind=DOMAIN_VERIFICATION_STAGE_KIND,
            stage=domain,
            required=True,
        ),
    )
    pipeline = VerificationPipeline(registry=verification_stage_registry(registrations))
    result = await pipeline.verify(_candidate())
    assert result.disposition is VerificationDisposition.CHALLENGED
    semantic_record = next(
        record for record in result.stage_records
        if record.stage == SEMANTIC_VERIFICATION_STAGE_KIND
    )
    assert semantic_record.outcome is VerificationStageOutcome.CHALLENGED
    assert semantic_record.challenge is not None
    assert semantic_record.challenge.finding.code == "verification.semantic.rubric_resolution_mismatch"
