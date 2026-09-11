"""Scenario-owned identity hypothesis composition helpers."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.application.identity.service import (
    ProductIdentityHypothesisService,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.strategy import (
    DeterministicEvidenceIdentityHypothesisStrategy,
)
from platform_proofs.scenarios.verified_product_identification.application.ports.catalog_search import (
    SourceRecordFetchPort,
)


def build_product_identity_hypothesis_service(
    *,
    source_port: SourceRecordFetchPort,
) -> ProductIdentityHypothesisService:
    """Construct the canonical deterministic identity hypothesis service."""

    return ProductIdentityHypothesisService(
        strategy=DeterministicEvidenceIdentityHypothesisStrategy(),
        source_port=source_port,
    )
