"""Scenario-owned production pipeline composition root."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.application.clarification.composition import (
    build_clarification_requirement_selection_service,
)
from platform_proofs.scenarios.verified_product_identification.application.fusion.composition import (
    build_offer_candidate_fusion,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.composition import (
    build_product_identity_hypothesis_service,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.composition import (
    build_identity_hypothesis_evaluation_service,
)
from platform_proofs.scenarios.verified_product_identification.application.observability.ports import (
    MonotonicClockPort,
    ProductIdentificationObservationSink,
)
from platform_proofs.scenarios.verified_product_identification.application.observability.sinks import (
    NoOpProductIdentificationObservationSink,
)
from platform_proofs.scenarios.verified_product_identification.application.pipeline.contracts import (
    ClarificationRequirementSelectionPort,
    IdentityHypothesisEvaluationPort,
    MultiChannelRetrievalPort,
    OfferCandidateFusionPort,
    ProductIdentificationPipelineConfiguration,
    ProductIdentityHypothesisPort,
)
from platform_proofs.scenarios.verified_product_identification.application.pipeline.retrieval_request_builder import (
    DeterministicProductIdentificationRetrievalRequestBuilder,
    ProductIdentificationRetrievalRequestBuilder,
)
from platform_proofs.scenarios.verified_product_identification.application.pipeline.service import (
    ProductIdentificationPipelineService,
)
from platform_proofs.scenarios.verified_product_identification.application.pipeline.stage_timing import (
    SystemMonotonicClock,
)
from platform_proofs.scenarios.verified_product_identification.application.ports.catalog_search import (
    ExactIdentifierLookupPort,
    LexicalCandidateSearchPort,
    SourceRecordFetchPort,
    StructuredCandidateSearchPort,
    VectorCandidateSearchPort,
)
from platform_proofs.scenarios.verified_product_identification.application.retrieval.service import (
    MultiChannelRetrievalService,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.composition import (
    build_product_identification_verification_service,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.service import (
    ProductIdentificationVerificationService,
)


def build_product_identification_pipeline(
    *,
    exact_lookup: ExactIdentifierLookupPort,
    lexical_search: LexicalCandidateSearchPort,
    structured_search: StructuredCandidateSearchPort,
    vector_search: VectorCandidateSearchPort,
    source_port: SourceRecordFetchPort,
    observation_sink: ProductIdentificationObservationSink | None = None,
    clock: MonotonicClockPort | None = None,
    configuration: ProductIdentificationPipelineConfiguration | None = None,
    retrieval_service: MultiChannelRetrievalPort | None = None,
    retrieval_request_builder: ProductIdentificationRetrievalRequestBuilder | None = None,
    fusion_service: OfferCandidateFusionPort | None = None,
    identity_service: ProductIdentityHypothesisPort | None = None,
    identity_evaluation_service: IdentityHypothesisEvaluationPort | None = None,
    verification_service: ProductIdentificationVerificationService | None = None,
    clarification_service: ClarificationRequirementSelectionPort | None = None,
) -> ProductIdentificationPipelineService:
    """Wire canonical scenario services — provider ports supplied at integration boundary."""

    resolved_retrieval = (
        retrieval_service
        if retrieval_service is not None
        else MultiChannelRetrievalService(
            exact_lookup=exact_lookup,
            lexical_search=lexical_search,
            structured_search=structured_search,
            vector_search=vector_search,
        )
    )
    resolved_fusion = fusion_service if fusion_service is not None else build_offer_candidate_fusion()
    resolved_identity = (
        identity_service
        if identity_service is not None
        else build_product_identity_hypothesis_service(source_port=source_port)
    )
    resolved_evaluation = (
        identity_evaluation_service
        if identity_evaluation_service is not None
        else build_identity_hypothesis_evaluation_service()
    )
    resolved_verification = (
        verification_service
        if verification_service is not None
        else build_product_identification_verification_service()
    )
    resolved_clarification = (
        clarification_service
        if clarification_service is not None
        else build_clarification_requirement_selection_service()
    )
    resolved_sink = (
        observation_sink
        if observation_sink is not None
        else NoOpProductIdentificationObservationSink()
    )
    resolved_clock = clock if clock is not None else SystemMonotonicClock()
    resolved_configuration = configuration or ProductIdentificationPipelineConfiguration()
    resolved_builder = (
        retrieval_request_builder
        if retrieval_request_builder is not None
        else DeterministicProductIdentificationRetrievalRequestBuilder(
            configuration=resolved_configuration,
        )
    )

    return ProductIdentificationPipelineService(
        retrieval_service=resolved_retrieval,
        retrieval_request_builder=resolved_builder,
        fusion_service=resolved_fusion,
        identity_service=resolved_identity,
        identity_evaluation_service=resolved_evaluation,
        verification_service=resolved_verification,
        clarification_service=resolved_clarification,
        observation_sink=resolved_sink,
        clock=resolved_clock,
        configuration=resolved_configuration,
    )
