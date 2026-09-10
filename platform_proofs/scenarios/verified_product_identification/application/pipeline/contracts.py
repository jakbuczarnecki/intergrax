"""Public production pipeline boundary contracts (5C12)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from platform_proofs.scenarios.verified_product_identification.application.clarification.contracts import (
    ClarificationSelectionRequest,
    ClarificationSelectionResult,
)
from platform_proofs.scenarios.verified_product_identification.application.clarification.service import (
    ClarificationRequirementSelectionService,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.failures import (
    CatalogSearchFailure,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.identification_context import (
    ProductIdentificationQueryContext,
)
from platform_proofs.scenarios.verified_product_identification.application.fusion.contracts import (
    FusedOfferCandidateCollection,
    OfferCandidateFusionRequest,
)
from platform_proofs.scenarios.verified_product_identification.application.fusion.service import (
    OfferCandidateFusionService,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.contracts import (
    ProductIdentityHypothesisCollection,
    ProductIdentityHypothesisRequest,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.service import (
    ProductIdentityHypothesisService,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.contracts import (
    IdentityHypothesisEvaluationRequest,
    RankedIdentityHypothesisCollection,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.service import (
    IdentityHypothesisEvaluationService,
)
from platform_proofs.scenarios.verified_product_identification.application.observability.contracts import (
    ProductIdentificationInputOrigin,
    ProductIdentificationRunId,
    ProductIdentificationStage,
)
from platform_proofs.scenarios.verified_product_identification.application.observability.ports import (
    ProductIdentificationObservationSink,
    ProductIdentificationObservationSinkMode,
)
from platform_proofs.scenarios.verified_product_identification.application.retrieval.contracts import (
    MultiChannelRetrievalRequest,
    MultiChannelRetrievalResult,
)
from platform_proofs.scenarios.verified_product_identification.application.retrieval.service import (
    MultiChannelRetrievalService,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.contracts import (
    ProductIdentificationDecision,
    ProductIdentificationVerificationOutcome,
    ProductIdentificationVerificationRequest,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.service import (
    ProductIdentificationVerificationService,
)


class MultiChannelRetrievalPort(Protocol):
    def retrieve(self, request: MultiChannelRetrievalRequest) -> MultiChannelRetrievalResult:
        ...


class OfferCandidateFusionPort(Protocol):
    def fuse(self, request: OfferCandidateFusionRequest) -> FusedOfferCandidateCollection:
        ...


class ProductIdentityHypothesisPort(Protocol):
    def form_hypotheses(
        self,
        request: ProductIdentityHypothesisRequest,
    ) -> ProductIdentityHypothesisCollection:
        ...


class IdentityHypothesisEvaluationPort(Protocol):
    def evaluate(
        self,
        request: IdentityHypothesisEvaluationRequest,
    ) -> RankedIdentityHypothesisCollection:
        ...


class ProductIdentificationVerificationPort(Protocol):
    def run(
        self,
        request: ProductIdentificationVerificationRequest,
    ) -> ProductIdentificationVerificationOutcome:
        ...


class ClarificationRequirementSelectionPort(Protocol):
    def select(self, request: ClarificationSelectionRequest) -> ClarificationSelectionResult:
        ...


@dataclass(frozen=True, slots=True)
class ProductIdentificationPipelineConfiguration:
    fusion_candidate_limit: int = 20
    identity_max_candidates: int = 20
    max_observation_offer_refs: int = 20
    observation_sink_mode: ProductIdentificationObservationSinkMode = (
        ProductIdentificationObservationSinkMode.BEST_EFFORT
    )

    def __post_init__(self) -> None:
        if type(self.fusion_candidate_limit) is not int or self.fusion_candidate_limit <= 0:
            raise ValueError("fusion_candidate_limit must be a positive int")
        if type(self.identity_max_candidates) is not int or self.identity_max_candidates <= 0:
            raise ValueError("identity_max_candidates must be a positive int")
        if (
            type(self.max_observation_offer_refs) is not int
            or self.max_observation_offer_refs <= 0
        ):
            raise ValueError("max_observation_offer_refs must be a positive int")


@dataclass(frozen=True, slots=True)
class ProductIdentificationPipelineRequest:
    query_context: ProductIdentificationQueryContext
    retrieval_request: MultiChannelRetrievalRequest
    input_origin: ProductIdentificationInputOrigin = (
        ProductIdentificationInputOrigin.TYPED_QUERY_CONTEXT
    )
    run_id: ProductIdentificationRunId | None = None
    catalog_content_identity: str | None = None


@dataclass(frozen=True, slots=True)
class ProductIdentificationPipelineStageFailure:
    stage: ProductIdentificationStage
    catalog_failure: CatalogSearchFailure | None = None
    observation_sink_failed: bool = False


@dataclass(frozen=True, slots=True)
class ProductIdentificationPipelineResult:
    run_id: ProductIdentificationRunId
    decision: ProductIdentificationDecision | None
    clarification: ClarificationSelectionResult | None
    stage_failure: ProductIdentificationPipelineStageFailure | None

    def __post_init__(self) -> None:
        if self.stage_failure is not None:
            if self.decision is not None or self.clarification is not None:
                raise ValueError("stage_failure excludes business decision and clarification")
        else:
            if self.decision is None:
                raise ValueError("successful pipeline path requires decision")


# Re-export concrete service types for composition typing without widening protocols.
PipelineRetrievalService = MultiChannelRetrievalService
PipelineFusionService = OfferCandidateFusionService
PipelineIdentityService = ProductIdentityHypothesisService
PipelineIdentityEvaluationService = IdentityHypothesisEvaluationService
PipelineVerificationService = ProductIdentificationVerificationService
PipelineClarificationService = ClarificationRequirementSelectionService
PipelineObservationSink = ProductIdentificationObservationSink
