# © Artur Czarnecki. All rights reserved.

"""Verification pipeline builders for DS-E2E."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.decision_record import CandidateDecision
from intergrax.contracts.decision_verification import (
    VerificationStageOutcome,
    validate_verification_stage_kind,
    verification_stage_record,
)
from intergrax.contracts.decision_verification_stage import (
    VerificationStageExecutionClass,
    VerificationStageRegistration,
    verification_stage_registry,
)
from intergrax.contracts.semantic_verification import (
    ResolvedSemanticRubric,
    SemanticContentProvider,
    SemanticRubricRef,
    VerifierIndependenceMode,
    resolved_semantic_rubric,
    semantic_rubric_ref,
    semantic_verification_independence_config,
)
from intergrax.runtime.decision_verification import VerificationPipeline
from intergrax.runtime.decision_verification_composition import (
    DecisionVerificationPipelineBuildSpec,
    SemanticProductionStageSpec,
    ToolWiringEvalVerificationBridge,
    build_decision_verification_pipeline,
    compose_tool_wiring_eval_verification,
)
from intergrax.runtime.execution.inference_profile import InferenceProfileId

from testing_support.decision_e2e.payloads import (
    QualificationRecommendation,
    QualificationSemanticContent,
)


@dataclass(frozen=True, slots=True)
class RecommendationSemanticExtractor:
    def extract(self, candidate: CandidateDecision[QualificationSemanticContent]) -> str:
        return candidate.artifact.content.text


@dataclass(frozen=True, slots=True)
class RecommendationToSemanticAdapter:
    def extract(self, candidate: CandidateDecision[QualificationRecommendation]) -> str:
        content = candidate.artifact.content
        parts = (content.recommendation.strip(), content.rationale_summary.strip())
        return ". ".join(part for part in parts if part)


class DeterministicPassStage:
    kind: str = "structural"

    @property
    def execution_class(self) -> VerificationStageExecutionClass:
        return VerificationStageExecutionClass.DETERMINISTIC

    async def verify(self, candidate):
        from intergrax.contracts.decision_record import candidate_decision_ref

        return verification_stage_record(
            proposal_ref=candidate_decision_ref(candidate),
            stage=validate_verification_stage_kind(self.kind),
            outcome=VerificationStageOutcome.PASSED,
        )


@dataclass(frozen=True, slots=True)
class StaticRubricResolver:
    rubric: ResolvedSemanticRubric

    def is_available(self) -> bool:
        return True

    def resolve(self, ref: SemanticRubricRef) -> ResolvedSemanticRubric:
        if ref.rubric_id != self.rubric.ref.rubric_id:
            raise ValueError("rubric not found")
        return self.rubric


def _default_rubric(*, rubric_id: str, min_score: float) -> ResolvedSemanticRubric:
    ref = semantic_rubric_ref(rubric_id=rubric_id, version=1)
    return resolved_semantic_rubric(
        ref=ref,
        criteria=("answer must be explicit and bounded",),
        min_score=min_score,
        provenance_ref="decision_e2e_qualification",
    )


def build_pass_through_pipeline() -> VerificationPipeline[QualificationRecommendation]:
    stage = DeterministicPassStage()
    return VerificationPipeline(
        registry=verification_stage_registry(
            (
                VerificationStageRegistration(
                    kind=validate_verification_stage_kind(stage.kind),
                    stage=stage,
                    required=True,
                ),
            ),
        ),
    )


def build_semantic_verification_pipeline(
    *,
    tool_bridge: ToolWiringEvalVerificationBridge,
    rubric_id: str,
    min_score: float,
    producer_profile_id: str,
    verifier_profile_id: str,
) -> VerificationPipeline[QualificationRecommendation]:
    rubric = _default_rubric(rubric_id=rubric_id, min_score=min_score)
    resolver = StaticRubricResolver(rubric=rubric)
    return build_decision_verification_pipeline(
        DecisionVerificationPipelineBuildSpec(
            eval_bridge=tool_bridge,
            semantic=SemanticProductionStageSpec(
                rubric_ref=rubric.ref,
                rubric_resolver=resolver,
                content_provider=RecommendationToSemanticAdapter(),
                independence=semantic_verification_independence_config(
                    mode=VerifierIndependenceMode.INDEPENDENT,
                    producer_profile_id=InferenceProfileId(producer_profile_id),
                    verifier_profile_id=InferenceProfileId(verifier_profile_id),
                ),
            ),
        ),
    )


def compose_eval_bridge(tool_wiring) -> ToolWiringEvalVerificationBridge:
    return compose_tool_wiring_eval_verification(tool_wiring)
