"""Shared bounded fixtures for VPI production pipeline tests (5C12)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from platform_proofs.scenarios.verified_product_identification.application.contracts.failures import (
    CatalogSearchFailure,
    CatalogSearchFailureKind,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.identification_context import (
    MissingDistinguishingRequirement,
    MissingRequirementOrigin,
    NegativeAttributeConstraint,
    ProductIdentificationQueryContext,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.product_identification_query import (
    ProductIdentificationQuery,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.queries import (
    ExactIdentifierQuery,
    StructuredAttributeConstraint,
    StructuredConstraintOperator,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.results import (
    ExactIdentifierLookupResult,
    LexicalSearchResult,
    SourceRecordFetchResult,
    StructuredSearchResult,
    VectorSearchResult,
)
from platform_proofs.scenarios.verified_product_identification.application.domain import (
    ExactChannelScore,
    ProductCandidate,
    ProductIdentifier,
    ProductIdentifierType,
    ProductOfferId,
    ProductSourceRecord,
    RetrievalChannel,
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.application.fusion.contracts import (
    FusedOfferCandidateCollection,
    OfferCandidateFusionRequest,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.contracts import (
    IdentityContradiction,
    IdentityContradictionType,
    IdentityEvidence,
    IdentityEvidenceProvenance,
    IdentityEvidenceStrengthClass,
    IdentityEvidenceType,
    IdentityHypothesisMember,
    ProductIdentityHypothesis,
    ProductIdentityHypothesisCollection,
    ProductIdentityHypothesisRequest,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.contracts import (
    IdentityHypothesisEvaluationRequest,
    RankedIdentityHypothesisCollection,
)
from platform_proofs.scenarios.verified_product_identification.application.observability.contracts import (
    ProductIdentificationRunId,
)
from platform_proofs.scenarios.verified_product_identification.application.pipeline.contracts import (
    ProductIdentificationPipelineRequest,
)
from platform_proofs.scenarios.verified_product_identification.application.retrieval.contracts import (
    ExactChannelRetrievalOutcome,
    LexicalChannelRetrievalOutcome,
    MultiChannelRetrievalRequest,
    MultiChannelRetrievalResult,
    RetrievalChannelExecutionStatus,
    RetrievalExecutionSummary,
    StructuredChannelRetrievalOutcome,
    VectorChannelRetrievalOutcome,
)
from platform_proofs.scenarios.verified_product_identification.application.catalog.candidate_handoff import (
    collect_channel_candidates,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    source_ref_sort_key,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source_identity import (
    source_ref_set_sha256,
)
from platform_proofs.scenarios.verified_product_identification.application.fusion import (
    OfferChannelEvidence,
    reciprocal_rank_contribution,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.source_identity_fact import (
    SourceIdentityFact,
    SourceIdentityFactKind,
    SourceIdentityFactProvenance,
)

CATALOG_ID = "catalog-alpha"
OFFER_A = ProductOfferId("offer-a")
OFFER_B = ProductOfferId("offer-b")
OFFER_C = ProductOfferId("offer-c")
OFFER_D = ProductOfferId("offer-d")


def source_ref(offer_id: ProductOfferId) -> SourceRecordRef:
    return SourceRecordRef(offer_id=offer_id, catalog_id=CATALOG_ID)


def constraint(name: str, value: str) -> StructuredAttributeConstraint:
    return StructuredAttributeConstraint(
        attribute_name=name,
        operator=StructuredConstraintOperator.EQUALS,
        value=value,
    )


def channel_evidence(channel: RetrievalChannel, *, rank: int = 0) -> OfferChannelEvidence:
    return OfferChannelEvidence(
        channel=channel,
        channel_rank=rank,
        channel_score=None,
        reciprocal_rank_contribution=reciprocal_rank_contribution(rank=rank, rrf_k=60),
    )


def _provenance(left_ref: SourceRecordRef, right_ref: SourceRecordRef) -> IdentityEvidenceProvenance:
    ordered = tuple(sorted((left_ref, right_ref), key=source_ref_sort_key))
    return IdentityEvidenceProvenance(
        left_source_ref=ordered[0],
        right_source_ref=ordered[1],
        source_field="test|test",
        normalization_rule="test/v1",
    )


def evidence_row(
    left_ref: SourceRecordRef,
    right_ref: SourceRecordRef,
    *,
    evidence_type: IdentityEvidenceType,
    attribute_key: str,
    normalized_value: str,
    identifier_type: ProductIdentifierType | None = None,
) -> IdentityEvidence:
    ordered = tuple(sorted((left_ref, right_ref), key=source_ref_sort_key))
    return IdentityEvidence(
        evidence_type=evidence_type,
        source_refs=ordered,
        attribute_key=attribute_key,
        normalized_value=normalized_value,
        strength_class=IdentityEvidenceStrengthClass.STRONG,
        identifier_type=identifier_type,
        provenance=_provenance(ordered[0], ordered[1]),
    )


def pair_hypothesis_with_facts(
    *,
    mpn: str,
    capacity: str,
    interface: str,
    refs: tuple[SourceRecordRef, SourceRecordRef],
) -> ProductIdentityHypothesis:
    ref_a, ref_b = refs
    facts = (
        source_fact(ref_a, attribute_key="capacity", normalized_value=capacity),
        source_fact(ref_b, attribute_key="capacity", normalized_value=capacity),
        source_fact(ref_a, attribute_key="interface", normalized_value=interface),
        source_fact(ref_b, attribute_key="interface", normalized_value=interface),
        source_fact(
            ref_a,
            attribute_key="mpn",
            normalized_value=mpn,
            identifier_type=ProductIdentifierType.MPN,
        ),
    )
    return hypothesis((ref_a, ref_b), source_identity_facts=facts)


def mpn_hypothesis(
    mpn: str,
    *,
    capacity: str | None = None,
    interface: str | None = None,
    refs: tuple[SourceRecordRef, SourceRecordRef] | None = None,
) -> ProductIdentityHypothesis:
    ref_a, ref_b = refs or (source_ref(OFFER_A), source_ref(OFFER_B))
    evidence: list[IdentityEvidence] = [
        evidence_row(
            ref_a,
            ref_b,
            evidence_type=IdentityEvidenceType.MODEL_NUMBER_MATCH,
            attribute_key="mpn",
            normalized_value=mpn,
            identifier_type=ProductIdentifierType.MPN,
        )
    ]
    if capacity is not None:
        evidence.append(
            evidence_row(
                ref_a,
                ref_b,
                evidence_type=IdentityEvidenceType.STRUCTURED_ATTRIBUTE_MATCH,
                attribute_key="capacity",
                normalized_value=capacity,
            )
        )
    if interface is not None:
        evidence.append(
            evidence_row(
                ref_a,
                ref_b,
                evidence_type=IdentityEvidenceType.STRUCTURED_ATTRIBUTE_MATCH,
                attribute_key="interface",
                normalized_value=interface,
            )
        )
    return hypothesis(tuple([ref_a, ref_b]), evidence=tuple(evidence))


def hypothesis(
    member_refs: tuple[SourceRecordRef, ...],
    *,
    evidence: tuple[IdentityEvidence, ...] = (),
    contradictions: tuple[IdentityContradiction, ...] = (),
    source_identity_facts: tuple[SourceIdentityFact, ...] = (),
) -> ProductIdentityHypothesis:
    ordered_refs = tuple(sorted(member_refs, key=source_ref_sort_key))
    members = tuple(
        IdentityHypothesisMember(
            source_ref=member_ref,
            fused_rank=index,
            fusion_evidence=(channel_evidence(RetrievalChannel.EXACT),),
        )
        for index, member_ref in enumerate(ordered_refs)
    )
    return ProductIdentityHypothesis(
        hypothesis_id=source_ref_set_sha256(ordered_refs),
        members=members,
        evidence=evidence,
        contradictions=contradictions,
        source_identity_facts=source_identity_facts,
    )


def source_fact(
    member_ref: SourceRecordRef,
    *,
    attribute_key: str,
    normalized_value: str,
    identifier_type: ProductIdentifierType | None = None,
) -> SourceIdentityFact:
    kind = (
        SourceIdentityFactKind.IDENTIFIER
        if identifier_type is not None
        else SourceIdentityFactKind.STRUCTURED_ATTRIBUTE
    )
    return SourceIdentityFact(
        source_ref=member_ref,
        fact_kind=kind,
        attribute_key=attribute_key,
        normalized_value=normalized_value,
        identifier_type=identifier_type,
        provenance=SourceIdentityFactProvenance(
            source_field=f"test|{attribute_key}",
            normalization_rule="test/v1",
            source_value=normalized_value,
        ),
    )


def exact_candidate(offer_id: ProductOfferId, *, rank: int) -> ProductCandidate:
    return ProductCandidate(
        offer_id=offer_id,
        channel=RetrievalChannel.EXACT,
        rank=rank,
        source_ref=source_ref(offer_id),
        channel_score=ExactChannelScore(
            matched_identifier=ProductIdentifier(
                identifier_type=ProductIdentifierType.GTIN,
                value="8806096660507",
            )
        ),
    )


def build_retrieval_result(
    *,
    exact: tuple[ProductCandidate, ...] = (),
    lexical: tuple[ProductCandidate, ...] = (),
    structured: tuple[ProductCandidate, ...] = (),
    vector: tuple[ProductCandidate, ...] = (),
    exact_status: RetrievalChannelExecutionStatus = RetrievalChannelExecutionStatus.SUCCESS,
    vector_status: RetrievalChannelExecutionStatus = RetrievalChannelExecutionStatus.SKIPPED,
    vector_failure: CatalogSearchFailure | None = None,
) -> MultiChannelRetrievalResult:
    exact_outcome = ExactChannelRetrievalOutcome(
        status=exact_status,
        lookup_results=(ExactIdentifierLookupResult(candidates=exact),) if exact else (),
        candidates=exact,
        failure=None,
    )
    lexical_outcome = LexicalChannelRetrievalOutcome(
        status=RetrievalChannelExecutionStatus.SKIPPED,
        search_result=None,
        candidates=lexical,
    )
    if lexical:
        lexical_outcome = LexicalChannelRetrievalOutcome(
            status=RetrievalChannelExecutionStatus.SUCCESS,
            search_result=LexicalSearchResult(candidates=lexical),
            candidates=lexical,
        )
    structured_outcome = StructuredChannelRetrievalOutcome(
        status=RetrievalChannelExecutionStatus.SKIPPED,
        search_result=None,
        candidates=structured,
    )
    if structured:
        structured_outcome = StructuredChannelRetrievalOutcome(
            status=RetrievalChannelExecutionStatus.SUCCESS,
            search_result=StructuredSearchResult(candidates=structured),
            candidates=structured,
        )
    vector_outcome = VectorChannelRetrievalOutcome(
        status=vector_status,
        search_result=VectorSearchResult(candidates=vector) if vector else None,
        candidates=vector,
        failure=vector_failure,
    )
    batches = []
    for channel, items in (
        (RetrievalChannel.EXACT, exact),
        (RetrievalChannel.LEXICAL, lexical),
        (RetrievalChannel.STRUCTURED, structured),
        (RetrievalChannel.VECTOR, vector),
    ):
        if items:
            from platform_proofs.scenarios.verified_product_identification.application.domain.candidates import (
                ChannelCandidateBatch,
            )

            batches.append(ChannelCandidateBatch(channel=channel, candidates=items))
    candidates = collect_channel_candidates(*batches) if batches else collect_channel_candidates()
    attempted = sum(
        1
        for status in (
            exact_outcome.status,
            lexical_outcome.status,
            structured_outcome.status,
            vector_outcome.status,
        )
        if status is not RetrievalChannelExecutionStatus.SKIPPED
    )
    succeeded = sum(
        1
        for status in (
            exact_outcome.status,
            lexical_outcome.status,
            structured_outcome.status,
            vector_outcome.status,
        )
        if status is RetrievalChannelExecutionStatus.SUCCESS
    )
    failed = sum(
        1
        for status in (
            exact_outcome.status,
            lexical_outcome.status,
            structured_outcome.status,
            vector_outcome.status,
        )
        if status is RetrievalChannelExecutionStatus.FAILED
    )
    skipped = 4 - attempted
    summary = RetrievalExecutionSummary(
        channels_attempted=attempted,
        channels_succeeded=succeeded,
        channels_failed=failed,
        channels_skipped=skipped,
        exact_candidate_count=len(exact),
        lexical_candidate_count=len(lexical),
        structured_candidate_count=len(structured),
        vector_candidate_count=len(vector),
    )
    return MultiChannelRetrievalResult(
        exact=exact_outcome,
        lexical=lexical_outcome,
        structured=structured_outcome,
        vector=vector_outcome,
        candidates=candidates,
        execution_summary=summary,
    )


def default_gtin_identifier() -> ProductIdentifier:
    return ProductIdentifier(
        identifier_type=ProductIdentifierType.GTIN,
        value="8806096660507",
    )


def pipeline_request(
    query_context: ProductIdentificationQueryContext | None = None,
    *,
    search_text: str | None = None,
) -> ProductIdentificationPipelineRequest:
    ctx = query_context if query_context is not None else ProductIdentificationQueryContext()
    if (
        not ctx.requested_identifiers
        and not ctx.required_constraints
        and search_text is None
    ):
        ctx = ProductIdentificationQueryContext(
            requested_identifiers=(default_gtin_identifier(),),
            negative_constraints=ctx.negative_constraints,
            missing_user_distinguishing_requirements=ctx.missing_user_distinguishing_requirements,
            soft_preferences=ctx.soft_preferences,
        )
    query = ProductIdentificationQuery(
        verification_context=ctx,
        search_text=search_text,
    )
    return ProductIdentificationPipelineRequest(
        query=query,
        run_id=ProductIdentificationRunId(value="run-test-001"),
    )


@dataclass(frozen=True, slots=True)
class FixedRetrievalService:
    result: MultiChannelRetrievalResult

    def retrieve(self, request: MultiChannelRetrievalRequest) -> MultiChannelRetrievalResult:
        return self.result


@dataclass(frozen=True, slots=True)
class FixedIdentityService:
    collection: ProductIdentityHypothesisCollection

    def form_hypotheses(
        self,
        request: ProductIdentityHypothesisRequest,
    ) -> ProductIdentityHypothesisCollection:
        return self.collection


@dataclass
class MapSourcePort:
    records: dict[SourceRecordRef, str]

    def fetch(self, source_ref: SourceRecordRef) -> SourceRecordFetchResult:
        payload = self.records.get(source_ref)
        if payload is None:
            return SourceRecordFetchResult(record=None, failure=None)
        return SourceRecordFetchResult(
            record=ProductSourceRecord(
                offer_id=source_ref.offer_id,
                catalog_id=source_ref.catalog_id,
                record_payload_ref=payload,
            ),
            failure=None,
        )


def wdc_payload(
    *,
    offer_id: str,
    gtin: str | None = None,
    mpn: str | None = None,
    capacity: str | None = None,
    interface: str | None = None,
) -> str:
    body: dict[str, object] = {"id": offer_id}
    identifiers: list[dict[str, str]] = []
    if gtin is not None:
        identifiers.append({"/gtin13": gtin})
    if mpn is not None:
        identifiers.append({"mpn": mpn})
    if identifiers:
        body["identifiers"] = identifiers
    pairs: dict[str, str] = {}
    if capacity is not None:
        pairs["capacity"] = capacity
    if interface is not None:
        pairs["interface"] = interface
    if pairs:
        body["keyValuePairs"] = pairs
    return json.dumps(body)
