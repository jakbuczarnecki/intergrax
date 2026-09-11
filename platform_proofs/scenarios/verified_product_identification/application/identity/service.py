"""Product identity hypothesis service boundary."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
    source_ref_sort_key,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
    parse_wdc_source_offer_json,
)
from platform_proofs.scenarios.verified_product_identification.application.fusion.contracts import (
    FusedOfferCandidate,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.contracts import (
    ProductIdentityHypothesisCollection,
    ProductIdentityHypothesisRequest,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.errors import (
    IdentityEvidenceUnavailableError,
    IdentityHypothesisError,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.pair_evidence import (
    OfferPairIdentityAssessment,
    assess_offer_pair,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.profile import (
    SourceOfferIdentityProfile,
    build_identity_profile,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.strategy import (
    ProductIdentityHypothesisStrategy,
)
from platform_proofs.scenarios.verified_product_identification.application.ports.catalog_search import (
    SourceRecordFetchPort,
)


@dataclass(frozen=True, slots=True)
class ProductIdentityHypothesisService:
    """Validate bounded fused input, load sources once, derive pair evidence, group."""

    strategy: ProductIdentityHypothesisStrategy
    source_port: SourceRecordFetchPort

    def form_hypotheses(
        self,
        request: ProductIdentityHypothesisRequest,
    ) -> ProductIdentityHypothesisCollection:
        candidates = request.fused_candidates.candidates
        if not candidates:
            return ProductIdentityHypothesisCollection(hypotheses=())

        _validate_unique_source_refs(candidates)
        profiles = _load_profiles(candidates, source_port=self.source_port)
        pair_assessments = _assess_all_pairs(candidates, profiles=profiles)
        return self.strategy.group(candidates, profiles, pair_assessments)


def _validate_unique_source_refs(candidates: tuple[FusedOfferCandidate, ...]) -> None:
    seen: set[SourceRecordRef] = set()
    for candidate in candidates:
        if candidate.source_ref in seen:
            raise IdentityHypothesisError("duplicate source_ref in fused candidate set")
        seen.add(candidate.source_ref)


def _load_profiles(
    candidates: tuple[FusedOfferCandidate, ...],
    *,
    source_port: SourceRecordFetchPort,
) -> dict[SourceRecordRef, SourceOfferIdentityProfile]:
    profiles: dict[SourceRecordRef, SourceOfferIdentityProfile] = {}
    ordered_refs = sorted(
        (candidate.source_ref for candidate in candidates),
        key=source_ref_sort_key,
    )
    for source_ref in ordered_refs:
        fetch_result = source_port.fetch(source_ref)
        if fetch_result.failure is not None:
            raise IdentityEvidenceUnavailableError(fetch_result.failure.message)
        record = fetch_result.record
        if record is None:
            raise IdentityEvidenceUnavailableError(
                f"source record not found for {source_ref.offer_id.value}"
            )
        if record.offer_id != source_ref.offer_id:
            raise IdentityEvidenceUnavailableError("source record identity does not match reference")
        source_offer = parse_wdc_source_offer_json(record.record_payload_ref)
        profiles[source_ref] = build_identity_profile(source_offer, source_ref=source_ref)
    return profiles


def _assess_all_pairs(
    candidates: tuple[FusedOfferCandidate, ...],
    *,
    profiles: dict[SourceRecordRef, SourceOfferIdentityProfile],
) -> dict[tuple[SourceRecordRef, SourceRecordRef], OfferPairIdentityAssessment]:
    candidate_by_ref = {candidate.source_ref: candidate for candidate in candidates}
    ordered_refs = sorted(
        (candidate.source_ref for candidate in candidates),
        key=source_ref_sort_key,
    )
    assessments: dict[tuple[SourceRecordRef, SourceRecordRef], OfferPairIdentityAssessment] = {}
    for left_index, left_ref in enumerate(ordered_refs):
        for right_ref in ordered_refs[left_index + 1 :]:
            assessment = assess_offer_pair(
                profiles[left_ref],
                profiles[right_ref],
                left_fused=candidate_by_ref[left_ref],
                right_fused=candidate_by_ref[right_ref],
            )
            assessments[(assessment.left_source_ref, assessment.right_source_ref)] = assessment
    return assessments
