"""Query understanding: raw input → ProductIdentificationQuery (bounded unit matrix)."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from platform_proofs.scenarios.verified_product_identification.application.contracts.identification_context import (
    ProductIdentificationQueryContext,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.product_identification_query import (
    ProductIdentificationQuery,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.queries import (
    StructuredConstraintOperator,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductIdentifierType,
)
from platform_proofs.scenarios.verified_product_identification.application.observability.contracts import (
    ProductIdentificationInputOrigin,
)
from platform_proofs.scenarios.verified_product_identification.application.clarification.composition import (
    build_clarification_requirement_selection_service,
)
from platform_proofs.scenarios.verified_product_identification.application.fusion.composition import (
    build_offer_candidate_fusion,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.composition import (
    build_identity_hypothesis_evaluation_service,
)
from platform_proofs.scenarios.verified_product_identification.application.observability import (
    InMemoryProductIdentificationObservationSink,
    ProductIdentificationObservationSinkMode,
)
from platform_proofs.scenarios.verified_product_identification.application.pipeline import (
    ProductIdentificationPipelineConfiguration,
    ProductIdentificationPipelineService,
)
from platform_proofs.scenarios.verified_product_identification.application.pipeline.stage_timing import (
    SystemMonotonicClock,
)
from platform_proofs.scenarios.verified_product_identification.application.verification import (
    build_product_identification_verification_service,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.queries import (
    StructuredAttributeConstraint,
)
from platform_proofs.scenarios.verified_product_identification.application.query_understanding.contracts import (
    QuerySourceSpan,
)
from platform_proofs.scenarios.verified_product_identification.application.pipeline.retrieval_request_builder import (
    DeterministicProductIdentificationRetrievalRequestBuilder,
)
from platform_proofs.scenarios.verified_product_identification.application.query_understanding.composition import (
    build_product_identification_query_understanding_service,
)
from platform_proofs.scenarios.verified_product_identification.application.query_understanding.contracts import (
    MAX_RAW_QUERY_CHARS,
    ExtractionCertainty,
    ExtractedConstraintRecord,
    ExtractedIdentifierRecord,
    ProductIdentificationQueryUnderstandingResult,
    QueryInterpretationCandidate,
    QueryUnderstandingIssueCode,
    QueryUnderstandingStatus,
    RawProductIdentificationRequest,
)
from platform_proofs.scenarios.verified_product_identification.application.query_understanding.extractors import (
    DeterministicProductIdentifierExtractor,
    DeterministicStructuredConstraintExtractor,
)
from platform_proofs.scenarios.verified_product_identification.application.query_understanding.policies import (
    DeterministicQueryUnderstandingMergePolicy,
)
from platform_proofs.scenarios.verified_product_identification.application.query_understanding.service import (
    ProductIdentificationQueryUnderstandingService,
)
from platform_proofs.scenarios.verified_product_identification.application.query_understanding.application_service import (
    ProductIdentificationApplicationService,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.vpi_pipeline_test_support import (
    FixedIdentityService,
    FixedRetrievalService,
    build_retrieval_result,
    pipeline_request,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.contracts import (
    ProductIdentityHypothesisCollection,
)

pytestmark = pytest.mark.unit


def _service(**kwargs: object) -> ProductIdentificationQueryUnderstandingService:
    base = build_product_identification_query_understanding_service()
    if not kwargs:
        return base
    return ProductIdentificationQueryUnderstandingService(
        identifier_extractor=kwargs.get("identifier_extractor", base.identifier_extractor),
        structured_extractor=kwargs.get("structured_extractor", base.structured_extractor),
        merge_policy=kwargs.get("merge_policy", base.merge_policy),
        interpreter=kwargs.get("interpreter", base.interpreter),
    )


def test_raw_request_immutable() -> None:
    raw = RawProductIdentificationRequest(raw_text="hello")
    with pytest.raises(Exception):
        raw.raw_text = "x"  # type: ignore[misc]


def test_empty_raw_input_rejected() -> None:
    with pytest.raises(ValueError):
        RawProductIdentificationRequest(raw_text="   ")


def test_oversized_raw_input_rejected() -> None:
    with pytest.raises(ValueError):
        RawProductIdentificationRequest(raw_text="a" * (MAX_RAW_QUERY_CHARS + 1))


def test_gtin_labeled_extraction() -> None:
    result = _service().understand(RawProductIdentificationRequest(raw_text="GTIN 4006381333931"))
    assert result.query is not None
    assert len(result.query.verification_context.requested_identifiers) == 1
    assert result.query.verification_context.requested_identifiers[0].identifier_type is ProductIdentifierType.GTIN


def test_mpn_labeled_extraction() -> None:
    result = _service().understand(RawProductIdentificationRequest(raw_text="MPN: ABC-123"))
    assert result.query is not None
    ids = result.query.verification_context.requested_identifiers
    assert ids[0].identifier_type is ProductIdentifierType.MPN
    assert ids[0].value == "ABC-123"


def test_sku_labeled_extraction() -> None:
    result = _service().understand(RawProductIdentificationRequest(raw_text="SKU: X771"))
    assert result.query is not None
    assert result.query.verification_context.requested_identifiers[0].identifier_type is ProductIdentifierType.SKU


def test_product_id_labeled_extraction() -> None:
    result = _service().understand(RawProductIdentificationRequest(raw_text="Product ID: 12345"))
    assert result.query is not None
    assert (
        result.query.verification_context.requested_identifiers[0].identifier_type
        is ProductIdentifierType.PRODUCT_ID
    )
    assert result.query.verification_context.requested_identifiers[0].value == "12345"


def test_invalid_gtin_typed_issue_not_accepted_as_identifier() -> None:
    result = _service().understand(RawProductIdentificationRequest(raw_text="GTIN 123"))
    assert result.status is QueryUnderstandingStatus.SUCCESS
    assert result.query is not None
    assert result.query.verification_context.requested_identifiers == ()
    assert any(i.code is QueryUnderstandingIssueCode.INVALID_IDENTIFIER for i in result.issues)


def test_unlabeled_ambiguous_code_not_guessed() -> None:
    result = _service().understand(RawProductIdentificationRequest(raw_text="ABC-123"))
    assert result.query is not None
    assert result.query.verification_context.requested_identifiers == ()


def test_identifier_normalization_preserved() -> None:
    result = _service().understand(RawProductIdentificationRequest(raw_text="GTIN 400-638-133-3931"))
    rec = result.extraction.identifiers[0]
    assert rec.raw_value
    assert rec.normalized_value == "4006381333931"
    assert rec.normalization_rule == "gtin_exact_lookup"


def test_provenance_span_preserved() -> None:
    text = "GTIN 4006381333931"
    result = _service().understand(RawProductIdentificationRequest(raw_text=text))
    span = result.extraction.identifiers[0].source_span
    assert text[span.start_offset : span.end_offset].upper().startswith("GTIN")


def test_required_constraint_extraction() -> None:
    result = _service().understand(RawProductIdentificationRequest(raw_text="2TB NVMe"))
    ctx = result.query.verification_context
    names = {c.attribute_name for c in ctx.required_constraints}
    assert "capacity" in names
    assert "interface" in names


def test_negative_constraint_extraction() -> None:
    result = _service().understand(RawProductIdentificationRequest(raw_text="not SATA"))
    neg = result.query.verification_context.negative_constraints
    assert len(neg) == 1
    assert neg[0].attribute_name == "interface"
    assert neg[0].excluded_value == "SATA"


def test_soft_phrase_not_hardened() -> None:
    result = _service().understand(RawProductIdentificationRequest(raw_text="ideally 2TB"))
    ctx = result.query.verification_context
    assert not ctx.required_constraints
    assert len(ctx.soft_preferences) == 1
    assert ctx.soft_preferences[0].value == "2TB"


def test_negative_phrase_not_converted_positive() -> None:
    result = _service().understand(RawProductIdentificationRequest(raw_text="not SATA"))
    required = {c.attribute_name for c in result.query.verification_context.required_constraints}
    assert "interface" not in required


def test_search_text_preserves_semantics() -> None:
    raw = "I need the Samsung 990 Pro with 2TB, not the 1TB version"
    result = _service().understand(RawProductIdentificationRequest(raw_text=raw))
    assert result.query is not None
    assert result.query.search_text == raw


def test_whitespace_normalization_deterministic() -> None:
    result = _service().understand(RawProductIdentificationRequest(raw_text="  2TB   NVMe  "))
    assert result.query is not None
    assert result.query.search_text == "2TB NVMe"


def test_duplicate_identifier_deduped() -> None:
    text = "GTIN 4006381333931 and GTIN 4006381333931"
    result = _service().understand(RawProductIdentificationRequest(raw_text=text))
    assert len(result.extraction.identifiers) == 1


def test_conflicting_identifiers_issue() -> None:
    text = "GTIN 4006381333931 GTIN 8806096660507"
    result = _service().understand(RawProductIdentificationRequest(raw_text=text))
    assert result.query is None
    assert any(
        i.code is QueryUnderstandingIssueCode.CONFLICTING_USER_CONSTRAINT for i in result.issues
    )


def test_conflicting_hard_constraint_issue() -> None:
    result = _service().understand(RawProductIdentificationRequest(raw_text="2TB and 1TB"))
    assert result.query is None
    assert any(
        i.code is QueryUnderstandingIssueCode.CONFLICTING_USER_CONSTRAINT for i in result.issues
    )


def test_no_actionable_semantics() -> None:
    result = _service().understand(RawProductIdentificationRequest(raw_text="???"))
    assert result.status is QueryUnderstandingStatus.REJECTED
    assert result.query is None
    assert any(i.code is QueryUnderstandingIssueCode.NO_ACTIONABLE_SEMANTICS for i in result.issues)


def test_typo_preserved_in_search_text() -> None:
    raw = "Samung 990 Pro 2TB"
    result = _service().understand(RawProductIdentificationRequest(raw_text=raw))
    assert result.query is not None
    assert result.query.search_text == raw


def test_qu_does_not_return_verification_outcome_type() -> None:
    result = _service().understand(RawProductIdentificationRequest(raw_text="GTIN 4006381333931"))
    assert isinstance(result, ProductIdentificationQueryUnderstandingResult)
    assert type(result).__name__ != "ProductIdentificationOutcome"


def test_same_input_deterministic() -> None:
    raw = RawProductIdentificationRequest(raw_text="GTIN 4006381333931 2TB NVMe")
    a = _service().understand(raw)
    b = _service().understand(raw)
    assert a == b


@dataclass(frozen=True, slots=True)
class _RecordingIdentifierExtractor:
    inner: DeterministicProductIdentifierExtractor
    called: bool = False

    def extract(self, raw_text: str):
        object.__setattr__(self, "called", True)
        return self.inner.extract(raw_text)


def test_custom_identifier_extractor_injectable() -> None:
    inner = DeterministicProductIdentifierExtractor()
    custom = _RecordingIdentifierExtractor(inner=inner)
    svc = _service(identifier_extractor=custom)
    svc.understand(RawProductIdentificationRequest(raw_text="GTIN 4006381333931"))
    assert custom.called


def test_custom_structured_extractor_injectable() -> None:
    custom = DeterministicStructuredConstraintExtractor()
    svc = _service(structured_extractor=custom)
    result = svc.understand(RawProductIdentificationRequest(raw_text="2TB"))
    assert result.query is not None


@dataclass(frozen=True, slots=True)
class _MaliciousInterpreter:
    def interpret(self, request: RawProductIdentificationRequest) -> QueryInterpretationCandidate:
        return QueryInterpretationCandidate(
            required_constraints=(
                ExtractedConstraintRecord(
                    constraint=StructuredAttributeConstraint(
                        attribute_name="capacity",
                        operator=StructuredConstraintOperator.EQUALS,
                        value="9TB",
                    ),
                    source_span=QuerySourceSpan(0, 1, "x"),
                    certainty=ExtractionCertainty.INFERRED,
                    raw_value="9TB",
                    normalized_value="9TB",
                ),
            )
        )


def test_interpreter_merge_and_deterministic_priority() -> None:
    svc = _service(interpreter=_MaliciousInterpreter())
    result = svc.understand(RawProductIdentificationRequest(raw_text="2TB NVMe"))
    caps = [c.value for c in result.query.verification_context.required_constraints if c.attribute_name == "capacity"]
    assert caps == ["2TB"]


def test_no_interpreter_required_for_deterministic() -> None:
    result = build_product_identification_query_understanding_service().understand(
        RawProductIdentificationRequest(raw_text="GTIN 4006381333931")
    )
    assert result.query is not None


def test_successful_result_contains_product_identification_query() -> None:
    result = _service().understand(RawProductIdentificationRequest(raw_text="GTIN 4006381333931"))
    assert isinstance(result.query, ProductIdentificationQuery)


def test_successful_query_passes_5c12_invariants() -> None:
    result = _service().understand(RawProductIdentificationRequest(raw_text="Samsung drive"))
    assert result.query is not None
    ProductIdentificationQuery(
        verification_context=result.query.verification_context,
        search_text=result.query.search_text,
    )


def test_output_contracts_hard_typed() -> None:
    result = _service().understand(RawProductIdentificationRequest(raw_text="2TB not SATA"))
    assert isinstance(result.query.verification_context, ProductIdentificationQueryContext)


def test_pipeline_accepts_query_understanding_output() -> None:
    understanding = _service().understand(
        RawProductIdentificationRequest(raw_text="GTIN 8806096660507")
    )
    assert understanding.query is not None
    builder = DeterministicProductIdentificationRetrievalRequestBuilder()
    built = builder.build(understanding.query)
    assert len(built.exact_queries) == 1


def test_retrieval_builder_gtin_from_extraction() -> None:
    understanding = _service().understand(
        RawProductIdentificationRequest(raw_text="GTIN 8806096660507")
    )
    built = DeterministicProductIdentificationRetrievalRequestBuilder().build(understanding.query)
    assert built.exact_queries[0].identifier.value == "8806096660507"


def test_retrieval_builder_structured_from_constraint() -> None:
    understanding = _service().understand(RawProductIdentificationRequest(raw_text="2TB"))
    built = DeterministicProductIdentificationRetrievalRequestBuilder().build(understanding.query)
    assert built.structured_query is not None


def test_raw_to_pipeline_e2e_typed_fakes() -> None:
    understanding_service = build_product_identification_query_understanding_service()
    configuration = ProductIdentificationPipelineConfiguration(
        observation_sink_mode=ProductIdentificationObservationSinkMode.BEST_EFFORT,
    )
    pipeline = ProductIdentificationPipelineService(
        retrieval_service=FixedRetrievalService(build_retrieval_result()),
        retrieval_request_builder=DeterministicProductIdentificationRetrievalRequestBuilder(
            configuration=configuration,
        ),
        fusion_service=build_offer_candidate_fusion(),
        identity_service=FixedIdentityService(ProductIdentityHypothesisCollection(hypotheses=())),
        identity_evaluation_service=build_identity_hypothesis_evaluation_service(),
        verification_service=build_product_identification_verification_service(),
        clarification_service=build_clarification_requirement_selection_service(),
        observation_sink=InMemoryProductIdentificationObservationSink(),
        clock=SystemMonotonicClock(),
        configuration=configuration,
    )
    app = ProductIdentificationApplicationService(
        query_understanding=understanding_service,
        pipeline=pipeline,
    )
    outcome = app.identify_from_raw(
        RawProductIdentificationRequest(raw_text="GTIN 8806096660507"),
        pipeline_request(),
    )
    assert outcome.pipeline is not None
    assert outcome.pipeline.decision is not None


def test_query_understanding_architecture_gates() -> None:
    repo = Path(__file__).resolve().parents[5]
    root = repo / "platform_proofs/scenarios/verified_product_identification/application/query_understanding"
    forbidden_fragments = (
        ".dataset.",
        "data_pack",
        "storage_bootstrap",
        "postgresql",
        "qdrant",
        "pgvector",
        ".proof.",
        "evaluator",
        "benchmark",
        "application.agent",
        "cluster_id",
        "openai",
        "anthropic",
        "google.generativeai",
        "gemini",
        "huggingface",
        "ollama",
    )
    violations: list[str] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module.casefold()
                for fragment in forbidden_fragments:
                    if fragment in module:
                        violations.append(f"{path.name}:{node.module}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name.casefold()
                    for fragment in forbidden_fragments:
                        if fragment in name:
                            violations.append(f"{path.name}:{alias.name}")
    assert violations == []
    sources = "".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))
    assert "cluster_id" not in sources


def test_qu_does_not_import_verification_module() -> None:
    repo = Path(__file__).resolve().parents[5]
    root = repo / "platform_proofs/scenarios/verified_product_identification/application/query_understanding"
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "verification" not in node.module
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "verification" not in alias.name


def test_pipeline_input_origin_raw_query() -> None:
    result = _service().understand(RawProductIdentificationRequest(raw_text="hello"))
    assert result.pipeline_input_origin is ProductIdentificationInputOrigin.RAW_QUERY
