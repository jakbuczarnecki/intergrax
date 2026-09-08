"""Bounded representation retrieval quality qualification."""

from platform_proofs.scenarios.verified_product_identification.qualification.bounded_representation.contracts import (
    BoundedRepresentationQualityReport,
    PerQueryRetrievalComparison,
    RepresentationVariant,
    VariantQualityGateResult,
)
from platform_proofs.scenarios.verified_product_identification.qualification.bounded_representation.evaluation import (
    compare_query_rankings,
    evaluate_quality_gate,
    evaluate_variant_rankings,
    select_winning_candidate,
)
from platform_proofs.scenarios.verified_product_identification.qualification.bounded_representation.truncation import (
    ProductRepresentationVariantPort,
    truncate_to_token_limit,
)

__all__ = (
    "BoundedRepresentationQualityReport",
    "PerQueryRetrievalComparison",
    "ProductRepresentationVariantPort",
    "RepresentationVariant",
    "VariantQualityGateResult",
    "compare_query_rankings",
    "evaluate_quality_gate",
    "evaluate_variant_rankings",
    "select_winning_candidate",
    "truncate_to_token_limit",
)
