"""Query / retrieval coherence and input-origin integrity (5C12-R1)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from platform_proofs.scenarios.verified_product_identification.application.clarification.composition import (
    build_clarification_requirement_selection_service,
)
from platform_proofs.scenarios.verified_product_identification.application.clarification.contracts import (
    ClarificationSelectionRequest,
    ClarificationSelectionResult,
)
from platform_proofs.scenarios.verified_product_identification.application.clarification.service import (
    ClarificationRequirementSelectionService,
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
    StructuredConstraintOperator,
)
from platform_proofs.scenarios.verified_product_identification.application.domain import (
    ProductIdentifier,
    ProductIdentifierType,
)
from platform_proofs.scenarios.verified_product_identification.application.fusion.composition import (
    build_offer_candidate_fusion,
)
from platform_proofs.scenarios.verified_product_identification.application.identity import (
    ProductIdentityHypothesisCollection,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.composition import (
    build_identity_hypothesis_evaluation_service,
)
from platform_proofs.scenarios.verified_product_identification.application.observability import (
    InMemoryProductIdentificationObservationSink,
    ProductIdentificationEventKind,
    ProductIdentificationInputOrigin,
    ProductIdentificationStage,
    QueryContextObservedPayload,
)
from platform_proofs.scenarios.verified_product_identification.application.pipeline.contracts import (
    ProductIdentificationPipelineConfiguration,
    ProductIdentificationPipelineRequest,
)
from platform_proofs.scenarios.verified_product_identification.application.pipeline.retrieval_request_builder import (
    DeterministicProductIdentificationRetrievalRequestBuilder,
)
from platform_proofs.scenarios.verified_product_identification.application.pipeline.service import (
    ProductIdentificationPipelineService,
)
from platform_proofs.scenarios.verified_product_identification.application.pipeline.stage_timing import (
    SystemMonotonicClock,
)
from platform_proofs.scenarios.verified_product_identification.application.retrieval.contracts import (
    MultiChannelRetrievalRequest,
    MultiChannelRetrievalResult,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.contracts import (
    ProductIdentificationVerificationOutcome,
    ProductIdentificationVerificationRequest,
)
from platform_proofs.scenarios.verified_product_identification.application.verification import (
    ProductIdentificationVerificationService,
    build_product_identification_verification_service,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.vpi_pipeline_test_support import (
    FixedIdentityService,
    FixedRetrievalService,
    build_retrieval_result,
    constraint,
    default_gtin_identifier,
    pipeline_request,
)

pytestmark = pytest.mark.unit

SEARCH_PHRASE = "Samsung 990 PRO 2TB"
GTIN_X = ProductIdentifier(identifier_type=ProductIdentifierType.GTIN, value="8806096660507")


def _builder() -> DeterministicProductIdentificationRetrievalRequestBuilder:
    return DeterministicProductIdentificationRetrievalRequestBuilder()


def test_query_contract_immutable() -> None:
    query = ProductIdentificationQuery(
        verification_context=ProductIdentificationQueryContext(
            requested_identifiers=(GTIN_X,),
        ),
    )
    assert query.__dataclass_params__.frozen is True


def test_query_rejects_empty_search_text() -> None:
    with pytest.raises(ValueError, match="search_text"):
        ProductIdentificationQuery(
            verification_context=ProductIdentificationQueryContext(
                requested_identifiers=(GTIN_X,),
            ),
            search_text="   ",
        )


def test_query_rejects_no_retrieval_semantics() -> None:
    with pytest.raises(ValueError, match="requires search_text"):
        ProductIdentificationQuery(verification_context=ProductIdentificationQueryContext())


def test_gtin_only_query_valid() -> None:
    ProductIdentificationQuery(
        verification_context=ProductIdentificationQueryContext(requested_identifiers=(GTIN_X,)),
    )


def test_structured_only_query_valid() -> None:
    ProductIdentificationQuery(
        verification_context=ProductIdentificationQueryContext(
            required_constraints=(
                constraint("capacity", "2TB"),
                constraint("interface", "NVMe"),
            ),
        ),
    )


def test_search_text_only_query_valid() -> None:
    ProductIdentificationQuery(
        verification_context=ProductIdentificationQueryContext(),
        search_text=SEARCH_PHRASE,
    )


def test_negative_only_invalid() -> None:
    with pytest.raises(ValueError):
        ProductIdentificationQuery(
            verification_context=ProductIdentificationQueryContext(
                negative_constraints=(
                    NegativeAttributeConstraint(
                        attribute_name="brand",
                        operator=StructuredConstraintOperator.EQUALS,
                        excluded_value="Acme",
                    ),
                ),
            ),
        )


def test_soft_preference_only_invalid() -> None:
    with pytest.raises(ValueError):
        ProductIdentificationQuery(
            verification_context=ProductIdentificationQueryContext(
                soft_preferences=(constraint("capacity", "2TB"),),
            ),
        )


def test_missing_requirement_only_invalid() -> None:
    with pytest.raises(ValueError):
        ProductIdentificationQuery(
            verification_context=ProductIdentificationQueryContext(
                missing_user_distinguishing_requirements=(
                    MissingDistinguishingRequirement(
                        attribute_name="capacity",
                        origin=MissingRequirementOrigin.USER,
                        requirement_id="req-1",
                    ),
                ),
            ),
        )


def test_builder_gtin_exact_coherence() -> None:
    query = ProductIdentificationQuery(
        verification_context=ProductIdentificationQueryContext(requested_identifiers=(GTIN_X,)),
    )
    built = _builder().build(query)
    assert len(built.exact_queries) == 1
    assert built.exact_queries[0].identifier == GTIN_X
    assert built.lexical_query is None
    assert built.vector_query is None


def test_builder_multiple_identifiers_deterministic_order() -> None:
    mpn = ProductIdentifier(identifier_type=ProductIdentifierType.MPN, value="MZ-V9P2T0")
    sku = ProductIdentifier(identifier_type=ProductIdentifierType.SKU, value="SKU-1")
    query = ProductIdentificationQuery(
        verification_context=ProductIdentificationQueryContext(
            requested_identifiers=(GTIN_X, mpn, sku),
        ),
    )
    first = _builder().build(query)
    second = _builder().build(query)
    assert first == second
    assert tuple(item.identifier for item in first.exact_queries) == (GTIN_X, mpn, sku)


def test_builder_structured_from_required_constraints() -> None:
    cap = constraint("capacity", "2TB")
    query = ProductIdentificationQuery(
        verification_context=ProductIdentificationQueryContext(required_constraints=(cap,)),
    )
    built = _builder().build(query)
    assert built.structured_query is not None
    assert built.structured_query.constraints == (cap,)


def test_builder_negative_not_in_structured() -> None:
    query = ProductIdentificationQuery(
        verification_context=ProductIdentificationQueryContext(
            requested_identifiers=(GTIN_X,),
            negative_constraints=(
                NegativeAttributeConstraint(
                    attribute_name="brand",
                    operator=StructuredConstraintOperator.EQUALS,
                    excluded_value="Other",
                ),
            ),
        ),
    )
    built = _builder().build(query)
    assert built.structured_query is None


def test_builder_soft_preferences_not_in_structured() -> None:
    query = ProductIdentificationQuery(
        verification_context=ProductIdentificationQueryContext(
            requested_identifiers=(GTIN_X,),
            soft_preferences=(constraint("capacity", "2TB"),),
        ),
    )
    built = _builder().build(query)
    assert built.structured_query is None


def test_builder_search_text_lexical_and_vector_identical() -> None:
    query = ProductIdentificationQuery(
        verification_context=ProductIdentificationQueryContext(requested_identifiers=(GTIN_X,)),
        search_text=SEARCH_PHRASE,
    )
    built = _builder().build(query)
    assert built.lexical_query is not None
    assert built.vector_query is not None
    assert built.lexical_query.query_text == SEARCH_PHRASE
    assert built.vector_query.query_text == SEARCH_PHRASE


def test_builder_search_text_absent_no_lexical_vector() -> None:
    built = _builder().build(
        ProductIdentificationQuery(
            verification_context=ProductIdentificationQueryContext(requested_identifiers=(GTIN_X,)),
        )
    )
    assert built.lexical_query is None
    assert built.vector_query is None


def test_builder_deterministic_for_same_query() -> None:
    query = ProductIdentificationQuery(
        verification_context=ProductIdentificationQueryContext(
            required_constraints=(constraint("capacity", "2TB"),),
        ),
        search_text=SEARCH_PHRASE,
    )
    assert _builder().build(query) == _builder().build(query)


def test_pipeline_request_has_no_retrieval_request_field() -> None:
    request = pipeline_request(ProductIdentificationQueryContext(requested_identifiers=(GTIN_X,)))
    assert not hasattr(request, "retrieval_request")
    assert not hasattr(request, "input_origin")
    assert not hasattr(request, "query_context")


@dataclass
class CapturingRetrievalService:
    last_request: MultiChannelRetrievalRequest | None = field(default=None, init=False)
    result: MultiChannelRetrievalResult = field(default_factory=build_retrieval_result)

    def retrieve(self, request: MultiChannelRetrievalRequest) -> MultiChannelRetrievalResult:
        self.last_request = request
        return self.result


@dataclass
class CapturingVerificationService:
    last_request: ProductIdentificationVerificationRequest | None = field(default=None, init=False)
    _delegate: ProductIdentificationVerificationService = field(
        default_factory=build_product_identification_verification_service
    )

    def run(
        self,
        request: ProductIdentificationVerificationRequest,
    ) -> ProductIdentificationVerificationOutcome:
        self.last_request = request
        return self._delegate.run(request)


@dataclass
class CapturingClarificationService:
    last_context: ProductIdentificationQueryContext | None = field(default=None, init=False)
    _delegate: ClarificationRequirementSelectionService = field(
        default_factory=build_clarification_requirement_selection_service
    )

    def select(self, request: ClarificationSelectionRequest) -> ClarificationSelectionResult:
        self.last_context = request.query_context
        return self._delegate.select(request)


def test_pipeline_retrieval_from_builder_verification_from_query_context() -> None:
    cap = constraint("capacity", "2TB")
    ctx = ProductIdentificationQueryContext(required_constraints=(cap,))
    query = ProductIdentificationQuery(verification_context=ctx)
    retrieval = CapturingRetrievalService()
    verification = CapturingVerificationService()
    clarification = CapturingClarificationService()
    configuration = ProductIdentificationPipelineConfiguration()
    sink = InMemoryProductIdentificationObservationSink()
    service = ProductIdentificationPipelineService(
        retrieval_service=retrieval,
        retrieval_request_builder=DeterministicProductIdentificationRetrievalRequestBuilder(
            configuration=configuration,
        ),
        fusion_service=build_offer_candidate_fusion(),
        identity_service=FixedIdentityService(ProductIdentityHypothesisCollection(hypotheses=())),
        identity_evaluation_service=build_identity_hypothesis_evaluation_service(),
        verification_service=verification,
        clarification_service=clarification,
        observation_sink=sink,
        clock=SystemMonotonicClock(),
        configuration=configuration,
    )
    service.run(
        ProductIdentificationPipelineRequest(
            query=query,
        )
    )
    assert retrieval.last_request is not None
    assert retrieval.last_request.structured_query is not None
    assert retrieval.last_request.structured_query.constraints == (cap,)
    assert verification.last_request is not None
    assert verification.last_request.query_context == ctx
    assert clarification.last_context == ctx


@dataclass(frozen=True, slots=True)
class VectorDisabledBuilder:
    configuration: ProductIdentificationPipelineConfiguration = field(
        default_factory=ProductIdentificationPipelineConfiguration
    )

    def build(self, query: ProductIdentificationQuery) -> MultiChannelRetrievalRequest:
        base = DeterministicProductIdentificationRetrievalRequestBuilder(
            configuration=self.configuration,
        ).build(query)
        return MultiChannelRetrievalRequest(
            exact_queries=base.exact_queries,
            lexical_query=base.lexical_query,
            structured_query=base.structured_query,
            vector_query=None,
        )


def test_custom_builder_injected_without_altering_verification_context() -> None:
    ctx = ProductIdentificationQueryContext(
        requested_identifiers=(GTIN_X,),
    )
    query = ProductIdentificationQuery(verification_context=ctx, search_text=SEARCH_PHRASE)
    retrieval = CapturingRetrievalService()
    verification = CapturingVerificationService()
    configuration = ProductIdentificationPipelineConfiguration()
    sink = InMemoryProductIdentificationObservationSink()
    service = ProductIdentificationPipelineService(
        retrieval_service=retrieval,
        retrieval_request_builder=VectorDisabledBuilder(configuration=configuration),
        fusion_service=build_offer_candidate_fusion(),
        identity_service=FixedIdentityService(ProductIdentityHypothesisCollection(hypotheses=())),
        identity_evaluation_service=build_identity_hypothesis_evaluation_service(),
        verification_service=verification,
        clarification_service=build_clarification_requirement_selection_service(),
        observation_sink=sink,
        clock=SystemMonotonicClock(),
        configuration=configuration,
    )
    service.run(ProductIdentificationPipelineRequest(query=query))
    assert retrieval.last_request is not None
    assert retrieval.last_request.vector_query is None
    assert verification.last_request is not None
    assert verification.last_request.query_context == ctx


def test_trace_reports_typed_query_context_only() -> None:
    sink = InMemoryProductIdentificationObservationSink()
    configuration = ProductIdentificationPipelineConfiguration()
    service = ProductIdentificationPipelineService(
        retrieval_service=FixedRetrievalService(build_retrieval_result()),
        retrieval_request_builder=DeterministicProductIdentificationRetrievalRequestBuilder(
            configuration=configuration,
        ),
        fusion_service=build_offer_candidate_fusion(),
        identity_service=FixedIdentityService(ProductIdentityHypothesisCollection(hypotheses=())),
        identity_evaluation_service=build_identity_hypothesis_evaluation_service(),
        verification_service=build_product_identification_verification_service(),
        clarification_service=build_clarification_requirement_selection_service(),
        observation_sink=sink,
        clock=SystemMonotonicClock(),
        configuration=configuration,
    )
    service.run(pipeline_request(ProductIdentificationQueryContext(requested_identifiers=(GTIN_X,))))
    origins = [
        item.payload.input_origin
        for item in sink.snapshot()
        if item.kind is ProductIdentificationEventKind.QUERY_CONTEXT
        and isinstance(item.payload, QueryContextObservedPayload)
    ]
    assert origins == [ProductIdentificationInputOrigin.TYPED_QUERY_CONTEXT]
