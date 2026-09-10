"""Map stage outputs to bounded typed observations."""

from __future__ import annotations

from collections import Counter

from platform_proofs.scenarios.verified_product_identification.application.clarification.contracts import (
    ClarificationSelectionResult,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.identification_context import (
    ProductIdentificationQueryContext,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.candidates import (
    RetrievalChannel,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.application.fusion.contracts import (
    FusedOfferCandidateCollection,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.contracts import (
    ProductIdentityHypothesis,
    ProductIdentityHypothesisCollection,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.contracts import (
    RankedIdentityHypothesisCollection,
)
from platform_proofs.scenarios.verified_product_identification.application.observability.contracts import (
    ClarificationObservedPayload,
    FusedOfferObserved,
    FusionObservedPayload,
    IdentityEvaluationObservedPayload,
    IdentityHypothesisSummaryObserved,
    IdentityHypothesesObservedPayload,
    ProductIdentificationInputOrigin,
    QueryContextObservedPayload,
    RetrievalChannelObservedPayload,
    TerminalObservedPayload,
    VerificationObservedPayload,
)
from platform_proofs.scenarios.verified_product_identification.application.pipeline.contracts import (
    ProductIdentificationPipelineConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.application.retrieval.contracts import (
    ExactChannelRetrievalOutcome,
    LexicalChannelRetrievalOutcome,
    MultiChannelRetrievalResult,
    RetrievalChannelExecutionStatus,
    StructuredChannelRetrievalOutcome,
    VectorChannelRetrievalOutcome,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.contracts import (
    ProductIdentificationDecision,
)


def build_query_context_payload(
    *,
    input_origin: ProductIdentificationInputOrigin,
    query_context: ProductIdentificationQueryContext,
    catalog_content_identity: str | None,
) -> QueryContextObservedPayload:
    return QueryContextObservedPayload(
        input_origin=input_origin,
        query_context=query_context,
        catalog_content_identity=catalog_content_identity,
    )


def _bounded_refs(
    candidates: tuple[SourceRecordRef, ...],
    *,
    limit: int,
) -> tuple[SourceRecordRef, ...]:
    return candidates[:limit]


def _channel_invoked_exact(outcome: ExactChannelRetrievalOutcome) -> bool:
    return outcome.status is not RetrievalChannelExecutionStatus.SKIPPED


def _channel_invoked_single(
    outcome: LexicalChannelRetrievalOutcome
    | StructuredChannelRetrievalOutcome
    | VectorChannelRetrievalOutcome,
) -> bool:
    return outcome.status is not RetrievalChannelExecutionStatus.SKIPPED


def build_retrieval_channel_payloads(
    retrieval: MultiChannelRetrievalResult,
    *,
    configuration: ProductIdentificationPipelineConfiguration,
    channel_durations_ns: dict[RetrievalChannel, int],
) -> tuple[RetrievalChannelObservedPayload, ...]:
    limit = configuration.max_observation_offer_refs
    specs: tuple[tuple[RetrievalChannel, ExactChannelRetrievalOutcome | LexicalChannelRetrievalOutcome | StructuredChannelRetrievalOutcome | VectorChannelRetrievalOutcome, bool], ...] = (
        (RetrievalChannel.EXACT, retrieval.exact, _channel_invoked_exact(retrieval.exact)),
        (RetrievalChannel.LEXICAL, retrieval.lexical, _channel_invoked_single(retrieval.lexical)),
        (RetrievalChannel.STRUCTURED, retrieval.structured, _channel_invoked_single(retrieval.structured)),
        (RetrievalChannel.VECTOR, retrieval.vector, _channel_invoked_single(retrieval.vector)),
    )
    rows: list[RetrievalChannelObservedPayload] = []
    for channel, outcome, invoked in specs:
        if not invoked:
            continue
        refs = _bounded_refs(
            tuple(candidate.source_ref for candidate in outcome.candidates),
            limit=limit,
        )
        rows.append(
            RetrievalChannelObservedPayload(
                channel=channel,
                invoked=True,
                status=outcome.status,
                candidate_count=len(outcome.candidates),
                bounded_offer_refs=refs,
                failure=outcome.failure,
                duration_ns=channel_durations_ns.get(channel, 0),
            )
        )
    return tuple(rows)


def build_fusion_payload(
    retrieval: MultiChannelRetrievalResult,
    fused: FusedOfferCandidateCollection,
    *,
    configuration: ProductIdentificationPipelineConfiguration,
) -> FusionObservedPayload:
    counts = (
        (RetrievalChannel.EXACT, len(retrieval.exact.candidates)),
        (RetrievalChannel.LEXICAL, len(retrieval.lexical.candidates)),
        (RetrievalChannel.STRUCTURED, len(retrieval.structured.candidates)),
        (RetrievalChannel.VECTOR, len(retrieval.vector.candidates)),
    )
    limit = configuration.max_observation_offer_refs
    fused_rows: list[FusedOfferObserved] = []
    for candidate in fused.candidates[:limit]:
        fused_rows.append(
            FusedOfferObserved(
                source_ref=candidate.source_ref,
                fused_rank=candidate.fused_rank,
                fusion_score=candidate.fusion_score,
                supporting_channel_count=candidate.supporting_channel_count,
                channel_evidence=candidate.evidence,
            )
        )
    return FusionObservedPayload(
        input_channel_candidate_counts=counts,
        merged_offer_count=len(fused.candidates),
        fused_offers=tuple(fused_rows),
    )


def _hypothesis_summary(hypothesis: ProductIdentityHypothesis) -> IdentityHypothesisSummaryObserved:
    evidence_counts = Counter(item.evidence_type.value for item in hypothesis.evidence)
    contradiction_counts = Counter(
        item.contradiction_type.value for item in hypothesis.contradictions
    )
    member_refs = tuple(member.source_ref for member in hypothesis.members)
    return IdentityHypothesisSummaryObserved(
        hypothesis_id=hypothesis.hypothesis_id,
        member_source_refs=member_refs,
        evidence_category_counts=tuple(sorted(evidence_counts.items())),
        contradiction_category_counts=tuple(sorted(contradiction_counts.items())),
    )


def build_identity_hypotheses_payload(
    collection: ProductIdentityHypothesisCollection,
) -> IdentityHypothesesObservedPayload:
    return IdentityHypothesesObservedPayload(
        hypotheses=tuple(_hypothesis_summary(item) for item in collection.hypotheses),
    )


def build_identity_evaluation_payload(
    ranked: RankedIdentityHypothesisCollection,
) -> IdentityEvaluationObservedPayload:
    order = tuple(item.hypothesis.hypothesis_id for item in ranked.hypotheses)
    return IdentityEvaluationObservedPayload(hypothesis_order=order, evaluated=ranked.hypotheses)


def build_verification_payload(decision: ProductIdentificationDecision) -> VerificationObservedPayload:
    return VerificationObservedPayload(
        hypothesis_verifications=decision.hypothesis_verifications,
        terminal_outcome=decision.outcome,
        reason_code=decision.decision_reason_code,
        verified_hypothesis_id=decision.verified_hypothesis_id,
        ambiguity_candidates=decision.ambiguity_candidates,
    )


def build_clarification_payload(
    clarification: ClarificationSelectionResult,
) -> ClarificationObservedPayload:
    return ClarificationObservedPayload(
        clarification_required=clarification.clarification_required,
        primary_requirement=clarification.primary_requirement,
        alternate_requirements=clarification.alternate_requirements,
        no_clarification_reason=clarification.no_clarification_reason,
    )


def build_terminal_payload(
    decision: ProductIdentificationDecision,
    clarification: ClarificationSelectionResult,
) -> TerminalObservedPayload:
    return TerminalObservedPayload(
        outcome=decision.outcome,
        reason_code=decision.decision_reason_code,
        verified_hypothesis_id=decision.verified_hypothesis_id,
        clarification_required=clarification.clarification_required,
    )
