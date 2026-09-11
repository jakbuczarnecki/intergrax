"""Identity hypothesis evaluation service boundary."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.identity.contracts import (
    ProductIdentityHypothesis,
    ProductIdentityHypothesisCollection,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.contradiction_evaluation import (
    build_contradiction_evaluation,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.contracts import (
    ContradictionRelationScope,
    EvidenceRelationScope,
    IdentityHypothesisEvaluationBundle,
    IdentityHypothesisEvaluationRequest,
    RankedIdentityHypothesisCollection,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.errors import (
    IdentityHypothesisEvaluationError,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.evidence_profile import (
    build_identity_evidence_profile,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.ranking_key import (
    build_ranking_key,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.scope import (
    classify_contradiction_scope,
    classify_evidence_scope,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.strategy import (
    IdentityHypothesisRankingStrategy,
)


@dataclass(frozen=True, slots=True)
class IdentityHypothesisEvaluationService:
    """Validate 5C8 output, derive profiles, invoke ranking policy, assign positions."""

    strategy: IdentityHypothesisRankingStrategy

    def evaluate(
        self,
        request: IdentityHypothesisEvaluationRequest,
    ) -> RankedIdentityHypothesisCollection:
        hypotheses = request.hypotheses.hypotheses
        if not hypotheses:
            return RankedIdentityHypothesisCollection(hypotheses=())

        _validate_collection(hypotheses)
        bundles = tuple(_build_bundle(hypothesis) for hypothesis in hypotheses)
        evaluated = self.strategy.rank(bundles)
        return RankedIdentityHypothesisCollection(hypotheses=evaluated)


def _validate_collection(hypotheses: tuple[ProductIdentityHypothesis, ...]) -> None:
    seen_ids: set[str] = set()
    for hypothesis in hypotheses:
        if hypothesis.hypothesis_id in seen_ids:
            raise IdentityHypothesisEvaluationError("duplicate hypothesis_id in collection")
        seen_ids.add(hypothesis.hypothesis_id)
        _validate_hypothesis_relations(hypothesis)


def _validate_hypothesis_relations(hypothesis: ProductIdentityHypothesis) -> None:
    member_refs = frozenset(member.source_ref for member in hypothesis.members)
    for evidence in hypothesis.evidence:
        scope = classify_evidence_scope(evidence, member_refs=member_refs)
        if scope is EvidenceRelationScope.INVALID:
            raise IdentityHypothesisEvaluationError(
                "evidence row unrelated to hypothesis members"
            )
    for contradiction in hypothesis.contradictions:
        scope = classify_contradiction_scope(contradiction, member_refs=member_refs)
        if scope is ContradictionRelationScope.INVALID:
            raise IdentityHypothesisEvaluationError(
                "contradiction row unrelated to hypothesis members"
            )


def _build_bundle(hypothesis: ProductIdentityHypothesis) -> IdentityHypothesisEvaluationBundle:
    evidence_profile = build_identity_evidence_profile(hypothesis)
    contradiction_evaluation = build_contradiction_evaluation(hypothesis)
    ranking_key = build_ranking_key(
        hypothesis,
        evidence_profile=evidence_profile,
        contradiction_evaluation=contradiction_evaluation,
    )
    return IdentityHypothesisEvaluationBundle(
        hypothesis=hypothesis,
        evidence_profile=evidence_profile,
        contradiction_evaluation=contradiction_evaluation,
        ranking_key=ranking_key,
    )
