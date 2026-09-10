"""Unit tests for product identity hypothesis formation (5C8)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from platform_proofs.scenarios.verified_product_identification.application.contracts.failures import (
    CatalogSearchFailure,
    CatalogSearchFailureKind,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.results import (
    SourceRecordFetchResult,
)
from platform_proofs.scenarios.verified_product_identification.application.domain import (
    LexicalChannelScore,
    ProductIdentifierType,
    ProductOfferId,
    ProductSourceProvenance,
    ProductSourceRecord,
    RetrievalChannel,
    SourceRecordRef,
    VectorChannelScore,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source_identity import (
    source_ref_set_sha256,
)
from platform_proofs.scenarios.verified_product_identification.application.fusion import (
    FusedOfferCandidate,
    FusedOfferCandidateCollection,
    OfferChannelEvidence,
    reciprocal_rank_contribution,
)
from platform_proofs.scenarios.verified_product_identification.application.identity import (
    IdentityContradictionType,
    IdentityEvidenceType,
    IdentityEvidenceUnavailableError,
    IdentityHypothesisConfiguration,
    IdentityHypothesisError,
    ProductIdentityHypothesisRequest,
    build_product_identity_hypothesis_service,
)

pytestmark = pytest.mark.unit

CATALOG_ID = "catalog-alpha"
CATALOG_BETA = "catalog-beta"
OFFER_A = ProductOfferId("offer-a")
OFFER_B = ProductOfferId("offer-b")
OFFER_C = ProductOfferId("offer-c")
OFFER_D = ProductOfferId("offer-d")


def _source_ref(
    offer_id: ProductOfferId,
    *,
    catalog_id: str = CATALOG_ID,
) -> SourceRecordRef:
    return SourceRecordRef(offer_id=offer_id, catalog_id=catalog_id)


def _wdc_payload(
    *,
    offer_id: str,
    brand: str | None = None,
    mpn: str | None = None,
    gtin: str | None = None,
    sku: str | None = None,
    product_id: str | None = None,
    capacity: str | None = None,
    color: str | None = None,
) -> str:
    payload: dict[str, object] = {"id": offer_id}
    if brand is not None:
        payload["brand"] = brand
    identifiers: list[dict[str, str]] = []
    if mpn is not None:
        identifiers.append({"mpn": mpn})
    if gtin is not None:
        identifiers.append({"/gtin13": gtin})
    if sku is not None:
        identifiers.append({"sku": sku})
    if product_id is not None:
        identifiers.append({"productId": product_id})
    if identifiers:
        payload["identifiers"] = identifiers
    key_value_pairs: dict[str, str] = {}
    if capacity is not None:
        key_value_pairs["capacity"] = capacity
    if color is not None:
        key_value_pairs["color"] = color
    if key_value_pairs:
        payload["keyValuePairs"] = key_value_pairs
    return json.dumps(payload)


def _channel_evidence(
    channel: RetrievalChannel,
    *,
    rank: int = 0,
    channel_score: LexicalChannelScore | VectorChannelScore | None = None,
) -> OfferChannelEvidence:
    return OfferChannelEvidence(
        channel=channel,
        channel_rank=rank,
        channel_score=channel_score,
        reciprocal_rank_contribution=reciprocal_rank_contribution(rank=rank, rrf_k=60),
    )


def _fused(
    offer_id: ProductOfferId,
    *,
    fused_rank: int,
    channels: tuple[RetrievalChannel, ...] = (),
    catalog_id: str = CATALOG_ID,
) -> FusedOfferCandidate:
    evidence = tuple(_channel_evidence(channel) for channel in channels)
    if not evidence:
        evidence = (_channel_evidence(RetrievalChannel.EXACT),)
    return FusedOfferCandidate(
        source_ref=_source_ref(offer_id, catalog_id=catalog_id),
        offer_id=offer_id,
        fused_rank=fused_rank,
        fusion_score=float(len(evidence)) / 61.0,
        supporting_channel_count=len(evidence),
        evidence=evidence,
    )


@dataclass
class _RecordingSourcePort:
    payloads: dict[SourceRecordRef, str]
    fetch_counts: dict[SourceRecordRef, int] = field(default_factory=dict)

    def fetch(self, source_ref: SourceRecordRef) -> SourceRecordFetchResult:
        self.fetch_counts[source_ref] = self.fetch_counts.get(source_ref, 0) + 1
        payload = self.payloads.get(source_ref)
        if payload is None:
            return SourceRecordFetchResult(
                record=None,
                failure=CatalogSearchFailure(
                    kind=CatalogSearchFailureKind.UNAVAILABLE,
                    message="missing source record",
                ),
            )
        return SourceRecordFetchResult(
            record=ProductSourceRecord(
                offer_id=source_ref.offer_id,
                record_payload_ref=payload,
                provenance=ProductSourceProvenance(catalog_id=source_ref.catalog_id),
            )
        )


def _service(payloads: dict[SourceRecordRef, str]) -> tuple[object, _RecordingSourcePort]:
    port = _RecordingSourcePort(payloads=payloads)
    return build_product_identity_hypothesis_service(source_port=port), port


def test_empty_fused_set_returns_empty_collection() -> None:
    service, _ = _service({})
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(candidates=()),
        )
    )
    assert result.hypotheses == ()


def test_singleton_candidate_returns_singleton_hypothesis() -> None:
    ref = _source_ref(OFFER_A)
    service, port = _service(
        {
            ref: _wdc_payload(offer_id=OFFER_A.value, brand="Samsung", mpn="MZ-V9P2T0"),
        }
    )
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(_fused(OFFER_A, fused_rank=0),)
            ),
        )
    )
    assert len(result.hypotheses) == 1
    assert len(result.hypotheses[0].members) == 1
    assert result.hypotheses[0].hypothesis_id == source_ref_set_sha256((ref,))
    assert port.fetch_counts[ref] == 1


def test_same_strong_identifier_supports_grouping() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    gtin = "8806096660507"
    service, _ = _service(
        {
            ref_a: _wdc_payload(offer_id=OFFER_A.value, gtin=gtin),
            ref_b: _wdc_payload(offer_id=OFFER_B.value, gtin=gtin),
        }
    )
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(
                    _fused(OFFER_A, fused_rank=0),
                    _fused(OFFER_B, fused_rank=1),
                )
            ),
        )
    )
    assert len(result.hypotheses) == 1
    assert {member.source_ref for member in result.hypotheses[0].members} == {ref_a, ref_b}
    assert any(
        item.evidence_type is IdentityEvidenceType.EXACT_IDENTIFIER_MATCH
        for item in result.hypotheses[0].evidence
    )


def test_conflicting_strong_identifiers_record_contradiction() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    service, _ = _service(
        {
            ref_a: _wdc_payload(offer_id=OFFER_A.value, gtin="8806096660507"),
            ref_b: _wdc_payload(offer_id=OFFER_B.value, gtin="0123456789012"),
        }
    )
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(
                    _fused(OFFER_A, fused_rank=0),
                    _fused(OFFER_B, fused_rank=1),
                )
            ),
        )
    )
    assert len(result.hypotheses) == 2
    contradictions = tuple(
        item
        for hypothesis in result.hypotheses
        for item in hypothesis.contradictions
        if item.contradiction_type is IdentityContradictionType.IDENTIFIER_CONFLICT
    )
    assert len(contradictions) == 2


def test_same_mpn_and_brand_support_grouping() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    service, _ = _service(
        {
            ref_a: _wdc_payload(
                offer_id=OFFER_A.value,
                brand="Samsung",
                mpn="MZ-V9P2T0",
            ),
            ref_b: _wdc_payload(
                offer_id=OFFER_B.value,
                brand="Samsung",
                mpn="MZ-V9P2T0",
            ),
        }
    )
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(
                    _fused(OFFER_A, fused_rank=0),
                    _fused(OFFER_B, fused_rank=1),
                )
            ),
        )
    )
    assert len(result.hypotheses) == 1


def test_same_mpn_conflicting_brand_blocks_grouping() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    service, _ = _service(
        {
            ref_a: _wdc_payload(offer_id=OFFER_A.value, brand="Samsung", mpn="MZ-V9P2T0"),
            ref_b: _wdc_payload(offer_id=OFFER_B.value, brand="Seagate", mpn="MZ-V9P2T0"),
        }
    )
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(
                    _fused(OFFER_A, fused_rank=0),
                    _fused(OFFER_B, fused_rank=1),
                )
            ),
        )
    )
    assert len(result.hypotheses) == 2
    assert any(
        item.contradiction_type is IdentityContradictionType.BRAND_CONFLICT
        for hypothesis in result.hypotheses
        for item in hypothesis.contradictions
    )


def test_missing_brand_is_unknown_not_conflict() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    service, _ = _service(
        {
            ref_a: _wdc_payload(offer_id=OFFER_A.value, mpn="MZ-V9P2T0"),
            ref_b: _wdc_payload(offer_id=OFFER_B.value, brand="Samsung", mpn="MZ-V9P2T0"),
        }
    )
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(
                    _fused(OFFER_A, fused_rank=0),
                    _fused(OFFER_B, fused_rank=1),
                )
            ),
        )
    )
    assert len(result.hypotheses) == 1
    assert not any(
        item.contradiction_type is IdentityContradictionType.BRAND_CONFLICT
        for hypothesis in result.hypotheses
        for item in hypothesis.contradictions
    )


def test_same_structured_capacity_supports_grouping_with_second_attribute() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    service, _ = _service(
        {
            ref_a: _wdc_payload(
                offer_id=OFFER_A.value,
                capacity="2TB",
                color="black",
            ),
            ref_b: _wdc_payload(
                offer_id=OFFER_B.value,
                capacity="2TB",
                color="black",
            ),
        }
    )
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(
                    _fused(OFFER_A, fused_rank=0),
                    _fused(OFFER_B, fused_rank=1),
                )
            ),
        )
    )
    assert len(result.hypotheses) == 1
    assert any(
        item.evidence_type is IdentityEvidenceType.STRUCTURED_ATTRIBUTE_MATCH
        for item in result.hypotheses[0].evidence
    )


def test_conflicting_structured_capacity_records_contradiction() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    service, _ = _service(
        {
            ref_a: _wdc_payload(offer_id=OFFER_A.value, capacity="2TB"),
            ref_b: _wdc_payload(offer_id=OFFER_B.value, capacity="1TB"),
        }
    )
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(
                    _fused(OFFER_A, fused_rank=0),
                    _fused(OFFER_B, fused_rank=1),
                )
            ),
        )
    )
    assert any(
        item.contradiction_type is IdentityContradictionType.STRUCTURED_ATTRIBUTE_CONFLICT
        for hypothesis in result.hypotheses
        for item in hypothesis.contradictions
    )


def test_unrelated_structured_keys_are_not_comparable() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    service, _ = _service(
        {
            ref_a: _wdc_payload(offer_id=OFFER_A.value, capacity="2TB"),
            ref_b: _wdc_payload(offer_id=OFFER_B.value, color="black"),
        }
    )
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(
                    _fused(OFFER_A, fused_rank=0),
                    _fused(OFFER_B, fused_rank=1),
                )
            ),
        )
    )
    assert len(result.hypotheses) == 2
    assert result.hypotheses[0].evidence == ()


def test_vector_only_similarity_does_not_force_merge() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    service, _ = _service(
        {
            ref_a: _wdc_payload(offer_id=OFFER_A.value, brand="Alpha"),
            ref_b: _wdc_payload(offer_id=OFFER_B.value, brand="Beta"),
        }
    )
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(
                    _fused(
                        OFFER_A,
                        fused_rank=0,
                        channels=(RetrievalChannel.VECTOR,),
                    ),
                    _fused(
                        OFFER_B,
                        fused_rank=1,
                        channels=(RetrievalChannel.VECTOR,),
                    ),
                )
            ),
        )
    )
    assert len(result.hypotheses) == 2
    weak = [
        item
        for hypothesis in result.hypotheses
        for item in hypothesis.evidence
        if item.evidence_type is IdentityEvidenceType.SEMANTIC_SUPPORT
    ]
    assert not weak


def test_lexical_only_similarity_does_not_force_merge() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    service, _ = _service(
        {
            ref_a: _wdc_payload(offer_id=OFFER_A.value),
            ref_b: _wdc_payload(offer_id=OFFER_B.value),
        }
    )
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(
                    _fused(
                        OFFER_A,
                        fused_rank=0,
                        channels=(RetrievalChannel.LEXICAL,),
                    ),
                    _fused(
                        OFFER_B,
                        fused_rank=1,
                        channels=(RetrievalChannel.LEXICAL,),
                    ),
                )
            ),
        )
    )
    assert len(result.hypotheses) == 2


def test_transitive_conflict_blocks_unsafe_merge() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    ref_c = _source_ref(OFFER_C)
    service, _ = _service(
        {
            ref_a: _wdc_payload(
                offer_id=OFFER_A.value,
                brand="Samsung",
                mpn="MZ-V9P2T0",
                capacity="2TB",
            ),
            ref_b: _wdc_payload(
                offer_id=OFFER_B.value,
                brand="Samsung",
                mpn="MZ-V9P2T0",
                capacity="2TB",
            ),
            ref_c: _wdc_payload(
                offer_id=OFFER_C.value,
                brand="Samsung",
                mpn="MZ-V9P1T0",
                capacity="1TB",
            ),
        }
    )
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(
                    _fused(OFFER_A, fused_rank=0),
                    _fused(OFFER_B, fused_rank=1),
                    _fused(OFFER_C, fused_rank=2),
                )
            ),
        )
    )
    grouped = {
        frozenset(member.source_ref for member in hypothesis.members)
        for hypothesis in result.hypotheses
    }
    assert frozenset({ref_a, ref_b}) in grouped
    assert frozenset({ref_c}) in grouped
    assert frozenset({ref_a, ref_b, ref_c}) not in grouped
    assert any(
        item.contradiction_type is IdentityContradictionType.MODEL_NUMBER_CONFLICT
        for hypothesis in result.hypotheses
        for item in hypothesis.contradictions
    )


def test_duplicate_source_refs_fail_closed() -> None:
    service, _ = _service({_source_ref(OFFER_A): _wdc_payload(offer_id=OFFER_A.value)})
    with pytest.raises(IdentityHypothesisError):
        service.form_hypotheses(
            ProductIdentityHypothesisRequest(
                fused_candidates=FusedOfferCandidateCollection(
                    candidates=(
                        _fused(OFFER_A, fused_rank=0),
                        _fused(OFFER_A, fused_rank=1),
                    )
                ),
            )
        )


def test_deterministic_hypothesis_id() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    payloads = {
        ref_a: _wdc_payload(offer_id=OFFER_A.value, gtin="8806096660507"),
        ref_b: _wdc_payload(offer_id=OFFER_B.value, gtin="8806096660507"),
    }
    service, _ = _service(payloads)
    request = ProductIdentityHypothesisRequest(
        fused_candidates=FusedOfferCandidateCollection(
            candidates=(
                _fused(OFFER_A, fused_rank=0),
                _fused(OFFER_B, fused_rank=1),
            )
        ),
    )
    first = service.form_hypotheses(request)
    second = service.form_hypotheses(request)
    assert first.hypotheses[0].hypothesis_id == source_ref_set_sha256((ref_a, ref_b))
    assert first == second


def test_deterministic_result_under_shuffled_input() -> None:
    from platform_proofs.scenarios.verified_product_identification.application.identity.pair_evidence import (
        assess_offer_pair,
    )
    from platform_proofs.scenarios.verified_product_identification.application.identity.profile import (
        build_identity_profile,
    )
    from platform_proofs.scenarios.verified_product_identification.application.identity.strategy import (
        DeterministicEvidenceIdentityHypothesisStrategy,
    )
    from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
        parse_wdc_source_offer_json,
    )

    refs = [_source_ref(OFFER_A), _source_ref(OFFER_B), _source_ref(OFFER_C)]
    payloads = {
        refs[0]: _wdc_payload(offer_id=OFFER_A.value, gtin="8806096660507"),
        refs[1]: _wdc_payload(offer_id=OFFER_B.value, gtin="8806096660507"),
        refs[2]: _wdc_payload(offer_id=OFFER_C.value, gtin="0123456789012"),
    }
    profiles = {
        ref: build_identity_profile(
            parse_wdc_source_offer_json(payloads[ref]),
            source_ref=ref,
        )
        for ref in refs
    }
    candidates = (
        _fused(OFFER_A, fused_rank=0),
        _fused(OFFER_B, fused_rank=1),
        _fused(OFFER_C, fused_rank=2),
    )
    candidate_by_ref = {candidate.source_ref: candidate for candidate in candidates}
    pair_assessments = {}
    for left_index, left_ref in enumerate(refs):
        for right_ref in refs[left_index + 1 :]:
            assessment = assess_offer_pair(
                profiles[left_ref],
                profiles[right_ref],
                left_fused=candidate_by_ref[left_ref],
                right_fused=candidate_by_ref[right_ref],
            )
            pair_assessments[(assessment.left_source_ref, assessment.right_source_ref)] = assessment

    strategy = DeterministicEvidenceIdentityHypothesisStrategy()
    ordered = strategy.group(candidates, profiles, pair_assessments)
    shuffled = strategy.group(
        (candidates[2], candidates[0], candidates[1]),
        profiles,
        pair_assessments,
    )
    assert ordered == shuffled


def test_source_fetched_once_per_unique_ref() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    service, port = _service(
        {
            ref_a: _wdc_payload(offer_id=OFFER_A.value, gtin="8806096660507"),
            ref_b: _wdc_payload(offer_id=OFFER_B.value, gtin="8806096660507"),
        }
    )
    service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(
                    _fused(OFFER_A, fused_rank=0),
                    _fused(OFFER_B, fused_rank=1),
                )
            ),
        )
    )
    assert port.fetch_counts[ref_a] == 1
    assert port.fetch_counts[ref_b] == 1


def test_missing_source_record_fails_explicitly() -> None:
    service, _ = _service({})
    with pytest.raises(IdentityEvidenceUnavailableError):
        service.form_hypotheses(
            ProductIdentityHypothesisRequest(
                fused_candidates=FusedOfferCandidateCollection(
                    candidates=(_fused(OFFER_A, fused_rank=0),)
                ),
            )
        )


def test_golden_fixture_samsung_ssd_hypotheses() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    ref_c = _source_ref(OFFER_C)
    service, _ = _service(
        {
            ref_a: _wdc_payload(
                offer_id=OFFER_A.value,
                brand="Samsung",
                mpn="MZ-V9P2T0",
                capacity="2TB",
            ),
            ref_b: _wdc_payload(
                offer_id=OFFER_B.value,
                brand="Samsung",
                mpn="MZ-V9P2T0",
                capacity="2TB",
            ),
            ref_c: _wdc_payload(
                offer_id=OFFER_C.value,
                brand="Samsung",
                mpn="MZ-V9P1T0",
                capacity="1TB",
            ),
        }
    )
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(
                    _fused(OFFER_A, fused_rank=0),
                    _fused(OFFER_B, fused_rank=1),
                    _fused(OFFER_C, fused_rank=2),
                )
            ),
        )
    )
    grouped = {
        frozenset(member.source_ref for member in hypothesis.members)
        for hypothesis in result.hypotheses
    }
    assert frozenset({ref_a, ref_b}) in grouped
    assert frozenset({ref_c}) in grouped
    cross_contradictions = [
        item
        for hypothesis in result.hypotheses
        for item in hypothesis.contradictions
        if ref_c in item.source_refs
    ]
    assert cross_contradictions


def test_same_gtin_across_catalogs_supports_grouping() -> None:
    ref_a = _source_ref(OFFER_A, catalog_id=CATALOG_ID)
    ref_b = _source_ref(OFFER_B, catalog_id=CATALOG_BETA)
    gtin = "8806096660507"
    service, _ = _service(
        {
            ref_a: _wdc_payload(offer_id=OFFER_A.value, gtin=gtin),
            ref_b: _wdc_payload(offer_id=OFFER_B.value, gtin=gtin),
        }
    )
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(
                    _fused(OFFER_A, fused_rank=0, catalog_id=CATALOG_ID),
                    _fused(OFFER_B, fused_rank=1, catalog_id=CATALOG_BETA),
                )
            ),
        )
    )
    assert len(result.hypotheses) == 1


def test_conflicting_gtin_across_catalogs_records_contradiction() -> None:
    ref_a = _source_ref(OFFER_A, catalog_id=CATALOG_ID)
    ref_b = _source_ref(OFFER_B, catalog_id=CATALOG_BETA)
    service, _ = _service(
        {
            ref_a: _wdc_payload(offer_id=OFFER_A.value, gtin="8806096660507"),
            ref_b: _wdc_payload(offer_id=OFFER_B.value, gtin="0123456789012"),
        }
    )
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(
                    _fused(OFFER_A, fused_rank=0, catalog_id=CATALOG_ID),
                    _fused(OFFER_B, fused_rank=1, catalog_id=CATALOG_BETA),
                )
            ),
        )
    )
    assert len(result.hypotheses) == 2
    assert any(
        item.contradiction_type is IdentityContradictionType.IDENTIFIER_CONFLICT
        for hypothesis in result.hypotheses
        for item in hypothesis.contradictions
    )


def test_different_mpn_same_brand_records_model_conflict() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    service, _ = _service(
        {
            ref_a: _wdc_payload(offer_id=OFFER_A.value, brand="Samsung", mpn="MZ-V9P2T0"),
            ref_b: _wdc_payload(offer_id=OFFER_B.value, brand="Samsung", mpn="MZ-V9P1T0"),
        }
    )
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(
                    _fused(OFFER_A, fused_rank=0),
                    _fused(OFFER_B, fused_rank=1),
                )
            ),
        )
    )
    assert len(result.hypotheses) == 2
    assert any(
        item.contradiction_type is IdentityContradictionType.MODEL_NUMBER_CONFLICT
        for hypothesis in result.hypotheses
        for item in hypothesis.contradictions
    )


def test_same_sku_across_catalogs_does_not_support_grouping() -> None:
    ref_a = _source_ref(OFFER_A, catalog_id=CATALOG_ID)
    ref_b = _source_ref(OFFER_B, catalog_id=CATALOG_BETA)
    service, _ = _service(
        {
            ref_a: _wdc_payload(offer_id=OFFER_A.value, sku="RETAIL-001"),
            ref_b: _wdc_payload(offer_id=OFFER_B.value, sku="RETAIL-001"),
        }
    )
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(
                    _fused(OFFER_A, fused_rank=0, catalog_id=CATALOG_ID),
                    _fused(OFFER_B, fused_rank=1, catalog_id=CATALOG_BETA),
                )
            ),
        )
    )
    assert len(result.hypotheses) == 2
    assert not any(
        item.evidence_type is IdentityEvidenceType.EXACT_IDENTIFIER_MATCH
        for hypothesis in result.hypotheses
        for item in hypothesis.evidence
    )


def test_different_sku_across_catalogs_does_not_contradict() -> None:
    ref_a = _source_ref(OFFER_A, catalog_id=CATALOG_ID)
    ref_b = _source_ref(OFFER_B, catalog_id=CATALOG_BETA)
    service, _ = _service(
        {
            ref_a: _wdc_payload(offer_id=OFFER_A.value, sku="RETAIL-001"),
            ref_b: _wdc_payload(offer_id=OFFER_B.value, sku="RETAIL-999"),
        }
    )
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(
                    _fused(OFFER_A, fused_rank=0, catalog_id=CATALOG_ID),
                    _fused(OFFER_B, fused_rank=1, catalog_id=CATALOG_BETA),
                )
            ),
        )
    )
    assert not any(
        item.contradiction_type is IdentityContradictionType.IDENTIFIER_CONFLICT
        for hypothesis in result.hypotheses
        for item in hypothesis.contradictions
    )


def test_same_product_id_across_catalogs_does_not_support_grouping() -> None:
    ref_a = _source_ref(OFFER_A, catalog_id=CATALOG_ID)
    ref_b = _source_ref(OFFER_B, catalog_id=CATALOG_BETA)
    service, _ = _service(
        {
            ref_a: _wdc_payload(offer_id=OFFER_A.value, product_id="pid-100"),
            ref_b: _wdc_payload(offer_id=OFFER_B.value, product_id="pid-100"),
        }
    )
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(
                    _fused(OFFER_A, fused_rank=0, catalog_id=CATALOG_ID),
                    _fused(OFFER_B, fused_rank=1, catalog_id=CATALOG_BETA),
                )
            ),
        )
    )
    assert len(result.hypotheses) == 2
    assert not any(
        item.evidence_type is IdentityEvidenceType.EXACT_IDENTIFIER_MATCH
        for hypothesis in result.hypotheses
        for item in hypothesis.evidence
    )


def test_different_product_id_across_catalogs_does_not_contradict() -> None:
    ref_a = _source_ref(OFFER_A, catalog_id=CATALOG_ID)
    ref_b = _source_ref(OFFER_B, catalog_id=CATALOG_BETA)
    service, _ = _service(
        {
            ref_a: _wdc_payload(offer_id=OFFER_A.value, product_id="pid-100"),
            ref_b: _wdc_payload(offer_id=OFFER_B.value, product_id="pid-200"),
        }
    )
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(
                    _fused(OFFER_A, fused_rank=0, catalog_id=CATALOG_ID),
                    _fused(OFFER_B, fused_rank=1, catalog_id=CATALOG_BETA),
                )
            ),
        )
    )
    assert not any(
        item.contradiction_type is IdentityContradictionType.IDENTIFIER_CONFLICT
        for hypothesis in result.hypotheses
        for item in hypothesis.contradictions
    )


def test_same_sku_with_vector_similarity_does_not_force_merge() -> None:
    ref_a = _source_ref(OFFER_A, catalog_id=CATALOG_ID)
    ref_b = _source_ref(OFFER_B, catalog_id=CATALOG_BETA)
    service, _ = _service(
        {
            ref_a: _wdc_payload(offer_id=OFFER_A.value, sku="RETAIL-001"),
            ref_b: _wdc_payload(offer_id=OFFER_B.value, sku="RETAIL-001"),
        }
    )
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(
                    _fused(
                        OFFER_A,
                        fused_rank=0,
                        channels=(RetrievalChannel.VECTOR,),
                        catalog_id=CATALOG_ID,
                    ),
                    _fused(
                        OFFER_B,
                        fused_rank=1,
                        channels=(RetrievalChannel.VECTOR,),
                        catalog_id=CATALOG_BETA,
                    ),
                )
            ),
        )
    )
    assert len(result.hypotheses) == 2


def test_gtin_support_not_cancelled_by_cross_catalog_sku_mismatch() -> None:
    ref_a = _source_ref(OFFER_A, catalog_id=CATALOG_ID)
    ref_b = _source_ref(OFFER_B, catalog_id=CATALOG_BETA)
    gtin = "8806096660507"
    service, _ = _service(
        {
            ref_a: _wdc_payload(offer_id=OFFER_A.value, gtin=gtin, sku="SKU-A"),
            ref_b: _wdc_payload(offer_id=OFFER_B.value, gtin=gtin, sku="SKU-B"),
        }
    )
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(
                    _fused(OFFER_A, fused_rank=0, catalog_id=CATALOG_ID),
                    _fused(OFFER_B, fused_rank=1, catalog_id=CATALOG_BETA),
                )
            ),
        )
    )
    assert len(result.hypotheses) == 1


def test_different_retailer_skus_groupable_via_gtin() -> None:
    ref_a = _source_ref(OFFER_A, catalog_id=CATALOG_ID)
    ref_b = _source_ref(OFFER_B, catalog_id=CATALOG_BETA)
    gtin = "8806096660507"
    service, _ = _service(
        {
            ref_a: _wdc_payload(
                offer_id=OFFER_A.value,
                gtin=gtin,
                sku="RETAILER-A-SKU",
            ),
            ref_b: _wdc_payload(
                offer_id=OFFER_B.value,
                gtin=gtin,
                sku="RETAILER-B-SKU",
            ),
        }
    )
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(
                    _fused(OFFER_A, fused_rank=0, catalog_id=CATALOG_ID),
                    _fused(OFFER_B, fused_rank=1, catalog_id=CATALOG_BETA),
                )
            ),
        )
    )
    assert len(result.hypotheses) == 1


def test_different_retailer_skus_groupable_via_mpn_and_brand() -> None:
    ref_a = _source_ref(OFFER_A, catalog_id=CATALOG_ID)
    ref_b = _source_ref(OFFER_B, catalog_id=CATALOG_BETA)
    service, _ = _service(
        {
            ref_a: _wdc_payload(
                offer_id=OFFER_A.value,
                brand="Samsung",
                mpn="MZ-V9P2T0",
                sku="RETAILER-A-SKU",
            ),
            ref_b: _wdc_payload(
                offer_id=OFFER_B.value,
                brand="Samsung",
                mpn="MZ-V9P2T0",
                sku="RETAILER-B-SKU",
            ),
        }
    )
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(
                    _fused(OFFER_A, fused_rank=0, catalog_id=CATALOG_ID),
                    _fused(OFFER_B, fused_rank=1, catalog_id=CATALOG_BETA),
                )
            ),
        )
    )
    assert len(result.hypotheses) == 1


def test_missing_identifiers_remain_unknown_not_conflicting() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    service, _ = _service(
        {
            ref_a: _wdc_payload(offer_id=OFFER_A.value),
            ref_b: _wdc_payload(offer_id=OFFER_B.value, gtin="8806096660507"),
        }
    )
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(
                    _fused(OFFER_A, fused_rank=0),
                    _fused(OFFER_B, fused_rank=1),
                )
            ),
        )
    )
    assert len(result.hypotheses) == 2
    assert not any(
        item.contradiction_type is IdentityContradictionType.IDENTIFIER_CONFLICT
        for hypothesis in result.hypotheses
        for item in hypothesis.contradictions
    )


def test_same_gtin_different_sku_same_catalog_groups() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    gtin = "8806096660507"
    service, _ = _service(
        {
            ref_a: _wdc_payload(offer_id=OFFER_A.value, gtin=gtin, sku="SKU-A"),
            ref_b: _wdc_payload(offer_id=OFFER_B.value, gtin=gtin, sku="SKU-B"),
        }
    )
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(
                    _fused(OFFER_A, fused_rank=0),
                    _fused(OFFER_B, fused_rank=1),
                )
            ),
        )
    )
    assert len(result.hypotheses) == 1


def test_different_gtin_same_sku_same_catalog_does_not_group() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    service, _ = _service(
        {
            ref_a: _wdc_payload(
                offer_id=OFFER_A.value,
                gtin="8806096660507",
                sku="RETAIL-001",
            ),
            ref_b: _wdc_payload(
                offer_id=OFFER_B.value,
                gtin="0123456789012",
                sku="RETAIL-001",
            ),
        }
    )
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(
                    _fused(OFFER_A, fused_rank=0),
                    _fused(OFFER_B, fused_rank=1),
                )
            ),
        )
    )
    assert len(result.hypotheses) == 2


def test_same_mpn_same_brand_different_sku_same_catalog_groups() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    service, _ = _service(
        {
            ref_a: _wdc_payload(
                offer_id=OFFER_A.value,
                brand="Samsung",
                mpn="MZ-V9P2T0",
                sku="SKU-A",
            ),
            ref_b: _wdc_payload(
                offer_id=OFFER_B.value,
                brand="Samsung",
                mpn="MZ-V9P2T0",
                sku="SKU-B",
            ),
        }
    )
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(
                    _fused(OFFER_A, fused_rank=0),
                    _fused(OFFER_B, fused_rank=1),
                )
            ),
        )
    )
    assert len(result.hypotheses) == 1


def test_same_mpn_conflicting_brand_same_sku_does_not_group() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    service, _ = _service(
        {
            ref_a: _wdc_payload(
                offer_id=OFFER_A.value,
                brand="Samsung",
                mpn="MZ-V9P2T0",
                sku="RETAIL-001",
            ),
            ref_b: _wdc_payload(
                offer_id=OFFER_B.value,
                brand="Seagate",
                mpn="MZ-V9P2T0",
                sku="RETAIL-001",
            ),
        }
    )
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(
                    _fused(OFFER_A, fused_rank=0),
                    _fused(OFFER_B, fused_rank=1),
                )
            ),
        )
    )
    assert len(result.hypotheses) == 2


def test_structured_strong_support_different_sku_same_catalog_groups() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    service, _ = _service(
        {
            ref_a: _wdc_payload(
                offer_id=OFFER_A.value,
                capacity="2TB",
                color="black",
                sku="SKU-A",
            ),
            ref_b: _wdc_payload(
                offer_id=OFFER_B.value,
                capacity="2TB",
                color="black",
                sku="SKU-B",
            ),
        }
    )
    result = service.form_hypotheses(
        ProductIdentityHypothesisRequest(
            fused_candidates=FusedOfferCandidateCollection(
                candidates=(
                    _fused(OFFER_A, fused_rank=0),
                    _fused(OFFER_B, fused_rank=1),
                )
            ),
        )
    )
    assert len(result.hypotheses) == 1


def test_sku_contradiction_preserved_but_nonblocking_for_gtin_grouping() -> None:
    from platform_proofs.scenarios.verified_product_identification.application.identity.pair_evidence import (
        assess_offer_pair,
        is_blocking_identity_contradiction,
        pair_has_grouping_eligibility,
    )
    from platform_proofs.scenarios.verified_product_identification.application.identity.profile import (
        build_identity_profile,
    )
    from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
        parse_wdc_source_offer_json,
    )

    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    gtin = "8806096660507"
    profiles = {
        ref_a: build_identity_profile(
            parse_wdc_source_offer_json(
                _wdc_payload(offer_id=OFFER_A.value, gtin=gtin, sku="SKU-A"),
            ),
            source_ref=ref_a,
        ),
        ref_b: build_identity_profile(
            parse_wdc_source_offer_json(
                _wdc_payload(offer_id=OFFER_B.value, gtin=gtin, sku="SKU-B"),
            ),
            source_ref=ref_b,
        ),
    }
    assessment = assess_offer_pair(
        profiles[ref_a],
        profiles[ref_b],
        left_fused=_fused(OFFER_A, fused_rank=0),
        right_fused=_fused(OFFER_B, fused_rank=1),
    )
    sku_contradictions = [
        item
        for item in assessment.contradictions
        if item.contradiction_type is IdentityContradictionType.IDENTIFIER_CONFLICT
        and item.identifier_type is ProductIdentifierType.SKU
    ]
    assert assessment.has_contradiction
    assert sku_contradictions
    assert all(not is_blocking_identity_contradiction(item) for item in sku_contradictions)
    assert pair_has_grouping_eligibility(
        assessment,
        left_profile=profiles[assessment.left_source_ref],
        right_profile=profiles[assessment.right_source_ref],
    )


def test_blocking_contradiction_classification_policy() -> None:
    from platform_proofs.scenarios.verified_product_identification.application.identity.pair_evidence import (
        assess_offer_pair,
        is_blocking_identity_contradiction,
    )
    from platform_proofs.scenarios.verified_product_identification.application.identity.profile import (
        build_identity_profile,
    )
    from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
        parse_wdc_source_offer_json,
    )

    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    gtin_assessment = assess_offer_pair(
        build_identity_profile(
            parse_wdc_source_offer_json(
                _wdc_payload(offer_id=OFFER_A.value, gtin="8806096660507"),
            ),
            source_ref=ref_a,
        ),
        build_identity_profile(
            parse_wdc_source_offer_json(
                _wdc_payload(offer_id=OFFER_B.value, gtin="0123456789012"),
            ),
            source_ref=ref_b,
        ),
        left_fused=_fused(OFFER_A, fused_rank=0),
        right_fused=_fused(OFFER_B, fused_rank=1),
    )
    gtin_conflicts = [
        item
        for item in gtin_assessment.contradictions
        if item.identifier_type is ProductIdentifierType.GTIN
    ]
    assert gtin_conflicts
    assert all(is_blocking_identity_contradiction(item) for item in gtin_conflicts)

    mpn_assessment = assess_offer_pair(
        build_identity_profile(
            parse_wdc_source_offer_json(
                _wdc_payload(offer_id=OFFER_A.value, brand="Samsung", mpn="MZ-V9P2T0"),
            ),
            source_ref=ref_a,
        ),
        build_identity_profile(
            parse_wdc_source_offer_json(
                _wdc_payload(offer_id=OFFER_B.value, brand="Samsung", mpn="MZ-V9P1T0"),
            ),
            source_ref=ref_b,
        ),
        left_fused=_fused(OFFER_A, fused_rank=0),
        right_fused=_fused(OFFER_B, fused_rank=1),
    )
    mpn_conflicts = [
        item
        for item in mpn_assessment.contradictions
        if item.contradiction_type is IdentityContradictionType.MODEL_NUMBER_CONFLICT
    ]
    assert mpn_conflicts
    assert all(is_blocking_identity_contradiction(item) for item in mpn_conflicts)

    brand_assessment = assess_offer_pair(
        build_identity_profile(
            parse_wdc_source_offer_json(
                _wdc_payload(offer_id=OFFER_A.value, brand="Samsung", mpn="MZ-V9P2T0"),
            ),
            source_ref=ref_a,
        ),
        build_identity_profile(
            parse_wdc_source_offer_json(
                _wdc_payload(offer_id=OFFER_B.value, brand="Seagate", mpn="MZ-V9P2T0"),
            ),
            source_ref=ref_b,
        ),
        left_fused=_fused(OFFER_A, fused_rank=0),
        right_fused=_fused(OFFER_B, fused_rank=1),
    )
    brand_conflicts = [
        item
        for item in brand_assessment.contradictions
        if item.contradiction_type is IdentityContradictionType.BRAND_CONFLICT
    ]
    assert brand_conflicts
    assert all(is_blocking_identity_contradiction(item) for item in brand_conflicts)
