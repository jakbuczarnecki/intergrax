"""Immutable production-path observation contracts for VPI (5C12)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from platform_proofs.scenarios.verified_product_identification.application.clarification.contracts import (
    ClarificationRequirement,
    NoClarificationReason,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.failures import (
    CatalogSearchFailure,
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
    OfferChannelEvidence,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.contracts import (
    EvaluatedIdentityHypothesis,
)
from platform_proofs.scenarios.verified_product_identification.application.retrieval.contracts import (
    RetrievalChannelExecutionStatus,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.contracts import (
    IdentityHypothesisVerification,
    ProductIdentificationDecisionReasonCode,
    ProductIdentificationOutcome,
)


class ProductIdentificationInputOrigin(StrEnum):
    """How the pipeline run received query semantics."""

    TYPED_QUERY_CONTEXT = "typed_query_context"
    RAW_QUERY = "raw_query"


class ProductIdentificationStage(StrEnum):
    """Closed observable pipeline stages."""

    QUERY_CONTEXT = "query_context"
    RETRIEVAL = "retrieval"
    FUSION = "fusion"
    IDENTITY_HYPOTHESIS = "identity_hypothesis"
    IDENTITY_EVALUATION = "identity_evaluation"
    VERIFICATION = "verification"
    CLARIFICATION = "clarification"
    TERMINAL = "terminal"


class ProductIdentificationEventKind(StrEnum):
    QUERY_CONTEXT = "query_context"
    RETRIEVAL_CHANNEL = "retrieval_channel"
    STAGE_TIMING = "stage_timing"
    FUSION = "fusion"
    IDENTITY_HYPOTHESES = "identity_hypotheses"
    IDENTITY_EVALUATION = "identity_evaluation"
    VERIFICATION = "verification"
    CLARIFICATION = "clarification"
    TERMINAL = "terminal"
    STAGE_FAILURE = "stage_failure"


@dataclass(frozen=True, slots=True)
class ProductIdentificationRunId:
    """Stable correlation identity for one complete identification execution."""

    value: str

    def __post_init__(self) -> None:
        if not self.value.strip():
            raise ValueError("ProductIdentificationRunId.value must be non-empty")


@dataclass(frozen=True, slots=True)
class QueryContextObservedPayload:
    input_origin: ProductIdentificationInputOrigin
    query_context: ProductIdentificationQueryContext
    catalog_content_identity: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalChannelObservedPayload:
    channel: RetrievalChannel
    invoked: bool
    status: RetrievalChannelExecutionStatus
    candidate_count: int
    bounded_offer_refs: tuple[SourceRecordRef, ...]
    failure: CatalogSearchFailure | None
    duration_ns: int

    def __post_init__(self) -> None:
        if type(self.candidate_count) is not int or self.candidate_count < 0:
            raise ValueError("candidate_count must be a non-negative int")
        if type(self.duration_ns) is not int or self.duration_ns < 0:
            raise ValueError("duration_ns must be a non-negative int")
        if not isinstance(self.bounded_offer_refs, tuple):
            raise TypeError("bounded_offer_refs must be a tuple")
        if self.status is RetrievalChannelExecutionStatus.FAILED:
            if self.failure is None:
                raise ValueError("failed channel requires failure evidence")
        elif self.failure is not None:
            raise ValueError("non-failed channel must not include failure")


@dataclass(frozen=True, slots=True)
class StageTimingObservedPayload:
    stage: ProductIdentificationStage
    duration_ns: int

    def __post_init__(self) -> None:
        if type(self.duration_ns) is not int or self.duration_ns < 0:
            raise ValueError("duration_ns must be a non-negative int")


@dataclass(frozen=True, slots=True)
class FusedOfferObserved:
    source_ref: SourceRecordRef
    fused_rank: int
    fusion_score: float
    supporting_channel_count: int
    channel_evidence: tuple[OfferChannelEvidence, ...]


@dataclass(frozen=True, slots=True)
class FusionObservedPayload:
    input_channel_candidate_counts: tuple[tuple[RetrievalChannel, int], ...]
    merged_offer_count: int
    fused_offers: tuple[FusedOfferObserved, ...]

    def __post_init__(self) -> None:
        if type(self.merged_offer_count) is not int or self.merged_offer_count < 0:
            raise ValueError("merged_offer_count must be a non-negative int")


@dataclass(frozen=True, slots=True)
class IdentityHypothesisSummaryObserved:
    hypothesis_id: str
    member_source_refs: tuple[SourceRecordRef, ...]
    evidence_category_counts: tuple[tuple[str, int], ...]
    contradiction_category_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class IdentityHypothesesObservedPayload:
    hypotheses: tuple[IdentityHypothesisSummaryObserved, ...]


@dataclass(frozen=True, slots=True)
class IdentityEvaluationObservedPayload:
    hypothesis_order: tuple[str, ...]
    evaluated: tuple[EvaluatedIdentityHypothesis, ...]


@dataclass(frozen=True, slots=True)
class VerificationObservedPayload:
    hypothesis_verifications: tuple[IdentityHypothesisVerification, ...]
    terminal_outcome: ProductIdentificationOutcome
    reason_code: ProductIdentificationDecisionReasonCode
    verified_hypothesis_id: str | None
    ambiguity_candidates: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ClarificationObservedPayload:
    clarification_required: bool
    primary_requirement: ClarificationRequirement | None
    alternate_requirements: tuple[ClarificationRequirement, ...]
    no_clarification_reason: NoClarificationReason | None


@dataclass(frozen=True, slots=True)
class TerminalObservedPayload:
    outcome: ProductIdentificationOutcome
    reason_code: ProductIdentificationDecisionReasonCode
    verified_hypothesis_id: str | None
    clarification_required: bool


@dataclass(frozen=True, slots=True)
class StageFailureObservedPayload:
    failed_stage: ProductIdentificationStage
    catalog_failure: CatalogSearchFailure | None
    observation_sink_failed: bool = False

    def __post_init__(self) -> None:
        if self.observation_sink_failed:
            return
        if self.catalog_failure is None:
            raise ValueError("stage failure requires catalog_failure unless sink failed")


ObservationPayload = (
    QueryContextObservedPayload
    | RetrievalChannelObservedPayload
    | StageTimingObservedPayload
    | FusionObservedPayload
    | IdentityHypothesesObservedPayload
    | IdentityEvaluationObservedPayload
    | VerificationObservedPayload
    | ClarificationObservedPayload
    | TerminalObservedPayload
    | StageFailureObservedPayload
)

_KIND_FOR_PAYLOAD: dict[type[ObservationPayload], ProductIdentificationEventKind] = {
    QueryContextObservedPayload: ProductIdentificationEventKind.QUERY_CONTEXT,
    RetrievalChannelObservedPayload: ProductIdentificationEventKind.RETRIEVAL_CHANNEL,
    StageTimingObservedPayload: ProductIdentificationEventKind.STAGE_TIMING,
    FusionObservedPayload: ProductIdentificationEventKind.FUSION,
    IdentityHypothesesObservedPayload: ProductIdentificationEventKind.IDENTITY_HYPOTHESES,
    IdentityEvaluationObservedPayload: ProductIdentificationEventKind.IDENTITY_EVALUATION,
    VerificationObservedPayload: ProductIdentificationEventKind.VERIFICATION,
    ClarificationObservedPayload: ProductIdentificationEventKind.CLARIFICATION,
    TerminalObservedPayload: ProductIdentificationEventKind.TERMINAL,
    StageFailureObservedPayload: ProductIdentificationEventKind.STAGE_FAILURE,
}

_STAGE_FOR_KIND: dict[ProductIdentificationEventKind, ProductIdentificationStage] = {
    ProductIdentificationEventKind.QUERY_CONTEXT: ProductIdentificationStage.QUERY_CONTEXT,
    ProductIdentificationEventKind.RETRIEVAL_CHANNEL: ProductIdentificationStage.RETRIEVAL,
    ProductIdentificationEventKind.STAGE_TIMING: ProductIdentificationStage.QUERY_CONTEXT,
    ProductIdentificationEventKind.FUSION: ProductIdentificationStage.FUSION,
    ProductIdentificationEventKind.IDENTITY_HYPOTHESES: ProductIdentificationStage.IDENTITY_HYPOTHESIS,
    ProductIdentificationEventKind.IDENTITY_EVALUATION: ProductIdentificationStage.IDENTITY_EVALUATION,
    ProductIdentificationEventKind.VERIFICATION: ProductIdentificationStage.VERIFICATION,
    ProductIdentificationEventKind.CLARIFICATION: ProductIdentificationStage.CLARIFICATION,
    ProductIdentificationEventKind.TERMINAL: ProductIdentificationStage.TERMINAL,
    ProductIdentificationEventKind.STAGE_FAILURE: ProductIdentificationStage.TERMINAL,
}


@dataclass(frozen=True, slots=True)
class ProductIdentificationObservation:
    run_id: ProductIdentificationRunId
    sequence: int
    stage: ProductIdentificationStage
    kind: ProductIdentificationEventKind
    payload: ObservationPayload

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("sequence must be a non-negative int")
        expected_kind = _KIND_FOR_PAYLOAD.get(type(self.payload))
        if expected_kind is None or expected_kind is not self.kind:
            raise ValueError("observation kind must match payload type")
        if self.kind is ProductIdentificationEventKind.STAGE_TIMING:
            timing = self.payload
            if not isinstance(timing, StageTimingObservedPayload):
                raise TypeError("stage timing payload required")
            if timing.stage is not self.stage:
                raise ValueError("stage timing stage must match observation stage")
        elif _STAGE_FOR_KIND.get(self.kind) is not self.stage:
            if self.kind is not ProductIdentificationEventKind.STAGE_FAILURE:
                raise ValueError("observation stage must match event kind")
