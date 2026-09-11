"""Identity hypothesis evidence evaluation — reranking and contradiction state."""

from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.composition import (
    build_identity_hypothesis_evaluation_service,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.contracts import (
    ContradictionRelationScope,
    EvaluatedIdentityHypothesis,
    EvidenceRelationScope,
    IdentityContradictionEvaluation,
    IdentityEvidenceProfile,
    IdentityHypothesisEvaluationBundle,
    IdentityHypothesisEvaluationRequest,
    IdentityHypothesisRankingKey,
    InternalPairCoverage,
    RankedIdentityHypothesisCollection,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.errors import (
    IdentityHypothesisEvaluationError,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.service import (
    IdentityHypothesisEvaluationService,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.strategy import (
    DeterministicEvidenceIdentityRankingStrategy,
    IdentityHypothesisRankingStrategy,
)

__all__ = (
    "ContradictionRelationScope",
    "DeterministicEvidenceIdentityRankingStrategy",
    "EvaluatedIdentityHypothesis",
    "EvidenceRelationScope",
    "IdentityContradictionEvaluation",
    "IdentityEvidenceProfile",
    "IdentityHypothesisEvaluationBundle",
    "IdentityHypothesisEvaluationError",
    "IdentityHypothesisEvaluationRequest",
    "IdentityHypothesisEvaluationService",
    "IdentityHypothesisRankingKey",
    "IdentityHypothesisRankingStrategy",
    "InternalPairCoverage",
    "RankedIdentityHypothesisCollection",
    "build_identity_hypothesis_evaluation_service",
)
