"""Composition root for query understanding."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.application.query_understanding.extractors import (
    DeterministicProductIdentifierExtractor,
    DeterministicStructuredConstraintExtractor,
)
from platform_proofs.scenarios.verified_product_identification.application.query_understanding.policies import (
    DeterministicQueryUnderstandingMergePolicy,
)
from platform_proofs.scenarios.verified_product_identification.application.query_understanding.service import (
    ProductIdentificationQueryUnderstandingService,
)


def build_product_identification_query_understanding_service(
    *,
    allow_structural_gtin: bool = True,
) -> ProductIdentificationQueryUnderstandingService:
    return ProductIdentificationQueryUnderstandingService(
        identifier_extractor=DeterministicProductIdentifierExtractor(
            allow_structural_gtin=allow_structural_gtin,
        ),
        structured_extractor=DeterministicStructuredConstraintExtractor(),
        merge_policy=DeterministicQueryUnderstandingMergePolicy(),
        interpreter=None,
    )
