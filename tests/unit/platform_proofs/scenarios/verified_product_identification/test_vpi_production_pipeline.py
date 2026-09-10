"""Production pipeline composition and observability tests (5C12)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from platform_proofs.scenarios.verified_product_identification.application.clarification.composition import (
    build_clarification_requirement_selection_service,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.failures import (
    CatalogSearchFailure,
    CatalogSearchFailureKind,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.identification_context import (
    HypothesisRejectionEvidence,
    MissingDistinguishingRequirement,
    MissingRequirementOrigin,
    ProductIdentificationQueryContext,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.queries import (
    ExactIdentifierQuery,
    VectorSearchQuery,
)
from platform_proofs.scenarios.verified_product_identification.application.domain import (
    ProductIdentifier,
    ProductIdentifierType,
    RetrievalChannel,
)
from platform_proofs.scenarios.verified_product_identification.application.fusion.composition import (
    build_offer_candidate_fusion,
)
from platform_proofs.scenarios.verified_product_identification.application.fusion.contracts import (
    FusedOfferCandidateCollection,
    OfferCandidateFusionRequest,
)
from platform_proofs.scenarios.verified_product_identification.application.fusion.service import (
    OfferCandidateFusionService,
)
from platform_proofs.scenarios.verified_product_identification.application.identity import (
    IdentityContradictionType,
    IdentityEvidenceType,
    ProductIdentityHypothesisCollection,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.composition import (
    build_product_identity_hypothesis_service,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.composition import (
    build_identity_hypothesis_evaluation_service,
)
from platform_proofs.scenarios.verified_product_identification.application.observability import (
    FailingProductIdentificationObservationSink,
    InMemoryProductIdentificationObservationSink,
    ObservationSinkError,
    ProductIdentificationEventKind,
    ProductIdentificationInputOrigin,
    ProductIdentificationObservationSinkMode,
    ProductIdentificationStage,
    QueryContextObservedPayload,
    RetrievalChannelObservedPayload,
    TerminalObservedPayload,
)
from platform_proofs.scenarios.verified_product_identification.application.pipeline import (
    ProductIdentificationPipelineConfiguration,
    ProductIdentificationPipelineService,
    build_product_identification_pipeline,
)
from platform_proofs.scenarios.verified_product_identification.application.pipeline.retrieval_request_builder import (
    DeterministicProductIdentificationRetrievalRequestBuilder,
)
from platform_proofs.scenarios.verified_product_identification.application.pipeline.stage_timing import (
    SystemMonotonicClock,
)
from platform_proofs.scenarios.verified_product_identification.application.retrieval import (
    MultiChannelRetrievalRequest,
    MultiChannelRetrievalResult,
    RetrievalChannelExecutionStatus,
)
from platform_proofs.scenarios.verified_product_identification.application.verification import (
    ProductIdentificationOutcome,
    build_product_identification_verification_service,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.contracts import (
    ProductIdentificationDecisionReasonCode,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.vpi_pipeline_test_support import (
    CATALOG_ID,
    OFFER_A,
    OFFER_B,
    OFFER_C,
    OFFER_D,
    FixedIdentityService,
    FixedRetrievalService,
    MapSourcePort,
    build_retrieval_result,
    constraint,
    default_gtin_identifier,
    evidence_row,
    exact_candidate,
    hypothesis,
    pipeline_request,
    source_fact,
    source_ref,
    wdc_payload,
)

pytestmark = pytest.mark.unit


def _pipeline(
    *,
    retrieval: FixedRetrievalService,
    identity: FixedIdentityService | None = None,
    sink: InMemoryProductIdentificationObservationSink,
    fusion_service: object | None = None,
    configuration: ProductIdentificationPipelineConfiguration | None = None,
) -> ProductIdentificationPipelineService:
    identity_service = identity or FixedIdentityService(ProductIdentityHypothesisCollection(hypotheses=()))
    resolved_configuration = configuration or ProductIdentificationPipelineConfiguration(
        observation_sink_mode=ProductIdentificationObservationSinkMode.BEST_EFFORT,
    )
    return ProductIdentificationPipelineService(
        retrieval_service=retrieval,
        retrieval_request_builder=DeterministicProductIdentificationRetrievalRequestBuilder(
            configuration=resolved_configuration,
        ),
        fusion_service=fusion_service or build_offer_candidate_fusion(),
        identity_service=identity_service,
        identity_evaluation_service=build_identity_hypothesis_evaluation_service(),
        verification_service=build_product_identification_verification_service(),
        clarification_service=build_clarification_requirement_selection_service(),
        observation_sink=sink,
        clock=SystemMonotonicClock(),
        configuration=resolved_configuration,
    )


def test_golden_verified_pipeline() -> None:
    ref_a, ref_b = source_ref(OFFER_A), source_ref(OFFER_B)
    h1 = hypothesis(
        (ref_a, ref_b),
        evidence=(
            evidence_row(
                ref_a,
                ref_b,
                evidence_type=IdentityEvidenceType.MODEL_NUMBER_MATCH,
                attribute_key="mpn",
                normalized_value="MZ-V9P2T0",
                identifier_type=ProductIdentifierType.MPN,
            ),
            evidence_row(
                ref_a,
                ref_b,
                evidence_type=IdentityEvidenceType.STRUCTURED_ATTRIBUTE_MATCH,
                attribute_key="capacity",
                normalized_value="2TB",
            ),
            evidence_row(
                ref_a,
                ref_b,
                evidence_type=IdentityEvidenceType.STRUCTURED_ATTRIBUTE_MATCH,
                attribute_key="interface",
                normalized_value="NVMe",
            ),
        ),
    )
    query = ProductIdentificationQueryContext(
        required_constraints=(constraint("capacity", "2TB"), constraint("interface", "NVMe")),
    )
    retrieval = build_retrieval_result(
        exact=(exact_candidate(OFFER_A, rank=0), exact_candidate(OFFER_B, rank=1)),
    )
    sink = InMemoryProductIdentificationObservationSink()
    service = _pipeline(
        retrieval=FixedRetrievalService(retrieval),
        identity=FixedIdentityService(ProductIdentityHypothesisCollection(hypotheses=(h1,))),
        sink=sink,
    )
    result = service.run(pipeline_request(query))
    assert result.decision is not None
    assert result.decision.outcome is ProductIdentificationOutcome.VERIFIED
    assert result.clarification is not None
    assert result.clarification.clarification_required is False
    terminals = [
        item
        for item in sink.snapshot()
        if item.kind is ProductIdentificationEventKind.TERMINAL
        and isinstance(item.payload, TerminalObservedPayload)
    ]
    assert len(terminals) == 1
    assert terminals[0].payload.outcome is ProductIdentificationOutcome.VERIFIED


def test_golden_ambiguous_pipeline() -> None:
    from tests.unit.platform_proofs.scenarios.verified_product_identification.vpi_pipeline_test_support import (
        pair_hypothesis_with_facts,
    )

    h1 = pair_hypothesis_with_facts(
        mpn="MZ-V9P1T0",
        capacity="1TB",
        interface="NVMe",
        refs=(source_ref(OFFER_A), source_ref(OFFER_B)),
    )
    h2 = pair_hypothesis_with_facts(
        mpn="MZ-V9P2T0",
        capacity="2TB",
        interface="NVMe",
        refs=(source_ref(OFFER_C), source_ref(OFFER_D)),
    )
    query = ProductIdentificationQueryContext(required_constraints=(constraint("interface", "NVMe"),))
    sink = InMemoryProductIdentificationObservationSink()
    service = _pipeline(
        retrieval=FixedRetrievalService(build_retrieval_result(exact=(exact_candidate(OFFER_A, rank=0),))),
        identity=FixedIdentityService(ProductIdentityHypothesisCollection(hypotheses=(h1, h2))),
        sink=sink,
    )
    result = service.run(pipeline_request(query))
    assert result.decision is not None
    assert result.decision.outcome is ProductIdentificationOutcome.AMBIGUOUS
    assert result.clarification is not None
    assert result.clarification.clarification_required is True
    assert result.clarification.primary_requirement is not None
    assert result.clarification.primary_requirement.attribute_name.casefold() == "capacity"


def test_golden_insufficient_pipeline() -> None:
    ref = source_ref(OFFER_A)
    h1 = hypothesis(
        (ref,),
        source_identity_facts=(
            source_fact(ref, attribute_key="capacity", normalized_value="2TB"),
        ),
    )
    query = ProductIdentificationQueryContext(
        required_constraints=(constraint("capacity", "2TB"), constraint("interface", "NVMe")),
    )
    sink = InMemoryProductIdentificationObservationSink()
    service = _pipeline(
        retrieval=FixedRetrievalService(build_retrieval_result()),
        identity=FixedIdentityService(ProductIdentityHypothesisCollection(hypotheses=(h1,))),
        sink=sink,
    )
    result = service.run(pipeline_request(query))
    assert result.decision is not None
    assert result.decision.outcome is ProductIdentificationOutcome.INSUFFICIENT_INFORMATION


def test_golden_no_match_pipeline() -> None:
    from tests.unit.platform_proofs.scenarios.verified_product_identification.vpi_pipeline_test_support import (
        mpn_hypothesis,
    )

    h1 = mpn_hypothesis("MZ-V9P1T0", capacity="1TB", interface="NVMe")
    query = ProductIdentificationQueryContext(required_constraints=(constraint("capacity", "2TB"),))
    sink = InMemoryProductIdentificationObservationSink()
    service = _pipeline(
        retrieval=FixedRetrievalService(build_retrieval_result()),
        identity=FixedIdentityService(ProductIdentityHypothesisCollection(hypotheses=(h1,))),
        sink=sink,
    )
    result = service.run(pipeline_request(query))
    assert result.decision is not None
    assert result.decision.outcome is ProductIdentificationOutcome.NO_MATCH
    assert result.decision.decision_contradicted_requirements or result.decision.decision_contradictions


def test_empty_retrieval_not_no_match() -> None:
    query = ProductIdentificationQueryContext()
    sink = InMemoryProductIdentificationObservationSink()
    service = _pipeline(
        retrieval=FixedRetrievalService(build_retrieval_result()),
        identity=FixedIdentityService(ProductIdentityHypothesisCollection(hypotheses=())),
        sink=sink,
    )
    result = service.run(pipeline_request(query))
    assert result.decision is not None
    assert result.decision.outcome is ProductIdentificationOutcome.INSUFFICIENT_INFORMATION


def test_channel_failure_not_empty_success() -> None:
    failure = CatalogSearchFailure(kind=CatalogSearchFailureKind.UNAVAILABLE, message="vector down")
    sink = InMemoryProductIdentificationObservationSink()
    service = _pipeline(
        retrieval=FixedRetrievalService(
            build_retrieval_result(
                exact=(exact_candidate(OFFER_A, rank=0),),
                vector_status=RetrievalChannelExecutionStatus.FAILED,
                vector_failure=failure,
            )
        ),
        sink=sink,
    )
    request = pipeline_request(
        ProductIdentificationQueryContext(
            requested_identifiers=(default_gtin_identifier(),),
        ),
        search_text="ssd",
    )
    result = service.run(request)
    channel_events = [
        item.payload
        for item in sink.snapshot()
        if item.kind is ProductIdentificationEventKind.RETRIEVAL_CHANNEL
        and isinstance(item.payload, RetrievalChannelObservedPayload)
    ]
    vector_events = [item for item in channel_events if item.channel is RetrievalChannel.VECTOR]
    assert vector_events
    assert vector_events[0].status is RetrievalChannelExecutionStatus.FAILED
    assert vector_events[0].failure is failure
    assert result.decision is not None


def test_retrieval_total_failure_short_circuits() -> None:
    failure = CatalogSearchFailure(kind=CatalogSearchFailureKind.UNAVAILABLE, message="all down")
    from platform_proofs.scenarios.verified_product_identification.application.retrieval.contracts import (
        ExactChannelRetrievalOutcome,
        VectorChannelRetrievalOutcome,
    )

    retrieval = build_retrieval_result(
        vector_status=RetrievalChannelExecutionStatus.FAILED,
        vector_failure=failure,
    )
    retrieval = MultiChannelRetrievalResult(
        exact=ExactChannelRetrievalOutcome(
            status=RetrievalChannelExecutionStatus.FAILED,
            lookup_results=(),
            candidates=(),
            failure=failure,
        ),
        lexical=retrieval.lexical,
        structured=retrieval.structured,
        vector=VectorChannelRetrievalOutcome(
            status=RetrievalChannelExecutionStatus.FAILED,
            search_result=None,
            candidates=(),
            failure=failure,
        ),
        candidates=retrieval.candidates,
        execution_summary=retrieval.execution_summary.__class__(
            channels_attempted=2,
            channels_succeeded=0,
            channels_failed=2,
            channels_skipped=2,
            exact_candidate_count=0,
            lexical_candidate_count=0,
            structured_candidate_count=0,
            vector_candidate_count=0,
        ),
    )
    sink = InMemoryProductIdentificationObservationSink()
    service = _pipeline(retrieval=FixedRetrievalService(retrieval), sink=sink)
    result = service.run(pipeline_request(search_text="x"))
    assert result.stage_failure is not None
    assert result.decision is None
    fusion_events = [
        item for item in sink.snapshot() if item.stage is ProductIdentificationStage.FUSION
    ]
    assert fusion_events == []


@dataclass
class CountingFusion:
    calls: int = 0
    _delegate: OfferCandidateFusionService = field(default_factory=build_offer_candidate_fusion)

    def fuse(self, request: OfferCandidateFusionRequest) -> FusedOfferCandidateCollection:
        self.calls += 1
        return self._delegate.fuse(request)


def test_fusion_invoked_after_retrieval() -> None:
    fusion = CountingFusion()
    sink = InMemoryProductIdentificationObservationSink()
    service = _pipeline(
        retrieval=FixedRetrievalService(build_retrieval_result(exact=(exact_candidate(OFFER_A, rank=0),))),
        identity=FixedIdentityService(ProductIdentityHypothesisCollection(hypotheses=())),
        sink=sink,
        fusion_service=fusion,
    )
    service.run(pipeline_request(ProductIdentificationQueryContext()))
    assert fusion.calls == 1


def test_observability_trace_invariants() -> None:
    sink = InMemoryProductIdentificationObservationSink()
    service = _pipeline(
        retrieval=FixedRetrievalService(build_retrieval_result()),
        sink=sink,
    )
    result = service.run(pipeline_request(ProductIdentificationQueryContext()))
    trace = sink.snapshot()
    assert all(item.run_id == result.run_id for item in trace)
    sequences = [item.sequence for item in trace]
    assert sequences == list(range(len(sequences)))
    terminals = [item for item in trace if item.kind is ProductIdentificationEventKind.TERMINAL]
    assert len(terminals) == 1
    query_events = [
        item.payload
        for item in trace
        if item.kind is ProductIdentificationEventKind.QUERY_CONTEXT
        and isinstance(item.payload, QueryContextObservedPayload)
    ]
    assert query_events
    assert query_events[0].input_origin is ProductIdentificationInputOrigin.TYPED_QUERY_CONTEXT
    assert ProductIdentificationInputOrigin.RAW_QUERY not in {
        payload.input_origin
        for payload in query_events
        if isinstance(payload, QueryContextObservedPayload)
    }


def test_required_sink_failure_not_silent_success() -> None:
    configuration = ProductIdentificationPipelineConfiguration(
        observation_sink_mode=ProductIdentificationObservationSinkMode.REQUIRED,
    )
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
        observation_sink=FailingProductIdentificationObservationSink(),
        clock=SystemMonotonicClock(),
        configuration=configuration,
    )
    result = service.run(pipeline_request(ProductIdentificationQueryContext()))
    assert result.stage_failure is not None
    assert result.stage_failure.observation_sink_failed is True
    assert result.decision is None


def test_pipeline_architecture_no_proof_imports() -> None:
    import ast
    from pathlib import Path

    repo = Path(__file__).resolve().parents[5]
    pipeline_root = repo / "platform_proofs/scenarios/verified_product_identification/application/pipeline"
    forbidden = (".proof.", "evaluator", "data_pack", "storage_bootstrap", ".dataset.")
    violations: list[str] = []
    for path in pipeline_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for fragment in forbidden:
                    if fragment in node.module:
                        violations.append(f"{path.name}:{node.module}")
    assert violations == []
