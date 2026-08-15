# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import pytest

from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter, VectorStoreHit
from intergrax.tools.providers.rag.bundle import (
    rag_preview_retrieval_contract,
    rag_retrieve_contract,
)
from intergrax.tools.providers.rag.contracts import RagRetrieveInput
from intergrax.tools.providers.rag.preview_service import rag_preview_retrieval
from intergrax.tools.providers.rag.service import _build_metadata_filter, perform_rag_retrieve
from intergrax.tools.providers.rag.source_scope_transport import (
    SOURCE_SCOPE_BLANK_ONLY,
    SOURCE_SCOPE_EMPTY,
    SOURCE_SCOPE_MALFORMED,
    RagRetrievalSourceScopeState,
    _ScopePresence,
    absent_rag_retrieval_source_scope,
    invalid_rag_retrieval_source_scope,
    parse_task_metadata_allowed_source_ids,
    rag_retrieval_source_scope,
    validated_rag_retrieval_source_scope,
)
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.runtime.nexus.tools.plan_context_invocation import build_rag_retrieve_input
from testing_support.builder import build_runtime_state_for_tests
from tests.unit.tools.providers.rag.test_rag_retrieve import (
    FakeEmbeddingManager,
    FakeVectorstoreManager,
    _document,
)

pytestmark = pytest.mark.unit


def _ctx(vectorstore: FakeVectorstoreManager | None = None) -> ToolWiringContext:
    return ToolWiringContext(
        vectorstore_manager=vectorstore or FakeVectorstoreManager([]),
        embedding_manager=FakeEmbeddingManager(),
        rag_profile=RagProfile(enable_rerank=False, route_mode="off", retriever_id="vector_similarity"),
    )


def _membership_allowed_values(metadata_filter: MetadataFilter | None) -> tuple[str, ...]:
    assert metadata_filter is not None
    assert metadata_filter.membership
    return metadata_filter.membership[0].allowed_values


def test_public_rag_retrieve_schema_excludes_allowed_source_ids() -> None:
    schema = rag_retrieve_contract().input_schema.model_json_schema()
    assert "allowed_source_ids" not in schema.get("properties", {})


def test_public_rag_preview_schema_excludes_allowed_source_ids() -> None:
    schema = rag_preview_retrieval_contract().input_schema.model_json_schema()
    assert "allowed_source_ids" not in schema.get("properties", {})


def test_public_rag_retrieve_input_still_works_unchanged() -> None:
    hits = [
        VectorStoreHit(
            vector_id="doc-1",
            document=_document("doc-1", "Policy baseline."),
            similarity_score=0.9,
            rank=1,
        )
    ]
    store = FakeVectorstoreManager(hits)
    out = perform_rag_retrieve(
        _ctx(store),
        RagRetrieveInput(query="policy", top_k=3, tenant_id="t1"),
    )
    assert out.used is True
    assert store.last_filter is None


def test_internal_single_source_scope_reaches_membership_filter() -> None:
    with rag_retrieval_source_scope(validated_rag_retrieval_source_scope(("source-a",))):
        metadata_filter = _build_metadata_filter(RagRetrieveInput(query="hello"))
    assert _membership_allowed_values(metadata_filter) == ("source-a",)


def test_internal_multi_source_scope_reaches_membership_filter() -> None:
    with rag_retrieval_source_scope(
        validated_rag_retrieval_source_scope(("source-b", "source-a"))
    ):
        metadata_filter = _build_metadata_filter(RagRetrieveInput(query="hello"))
    assert _membership_allowed_values(metadata_filter) == ("source-b", "source-a")


def test_no_internal_scope_has_no_membership_filter() -> None:
    metadata_filter = _build_metadata_filter(RagRetrieveInput(query="hello"))
    assert metadata_filter is None


@pytest.mark.parametrize(
    ("scope_state", "expected_reason"),
    [
        (invalid_rag_retrieval_source_scope(SOURCE_SCOPE_EMPTY), SOURCE_SCOPE_EMPTY),
        (invalid_rag_retrieval_source_scope(SOURCE_SCOPE_BLANK_ONLY), SOURCE_SCOPE_BLANK_ONLY),
        (invalid_rag_retrieval_source_scope(SOURCE_SCOPE_MALFORMED), SOURCE_SCOPE_MALFORMED),
    ],
)
def test_explicit_invalid_internal_scope_fails_closed(
    scope_state,
    expected_reason: str,
) -> None:
    store = FakeVectorstoreManager([])
    with rag_retrieval_source_scope(scope_state):
        out = perform_rag_retrieve(_ctx(store), RagRetrieveInput(query="policy", tenant_id="t1"))
    assert out.used is False
    assert out.reason == expected_reason
    assert store.query_calls == 0


def test_smuggled_public_allowed_source_ids_do_not_apply_membership() -> None:
    store = FakeVectorstoreManager([])
    payload = RagRetrieveInput.model_validate(
        {
            "query": "policy",
            "tenant_id": "t1",
            "allowed_source_ids": ["source-smuggled"],
        }
    )
    out = perform_rag_retrieve(_ctx(store), payload)
    assert out.used is False
    assert store.last_filter is None


def test_preview_retrieval_uses_internal_scope_not_public_field() -> None:
    store = FakeVectorstoreManager(
        [
            VectorStoreHit(
                vector_id="doc-1",
                document=_document("doc-1", "Scoped text"),
                similarity_score=0.9,
                rank=1,
            )
        ]
    )
    with rag_retrieval_source_scope(validated_rag_retrieval_source_scope(("source-a",))):
        out = rag_preview_retrieval(
            _ctx(store),
            RagRetrieveInput(query="policy", tenant_id="t1"),
        )
    assert out.used is True
    assert store.last_filter is not None
    assert _membership_allowed_values(store.last_filter) == ("source-a",)


def test_planner_rag_input_builder_never_supplies_source_ids() -> None:
    state = build_runtime_state_for_tests(run_id="run-planner-scope")
    state.request.message = "What is the policy?"
    built = build_rag_retrieve_input(state)
    dumped = built.model_dump()
    assert "allowed_source_ids" not in dumped
    assert "source_id" not in dumped
    assert "source_ids" not in dumped


def test_absent_internal_scope_does_not_fail_closed() -> None:
    with rag_retrieval_source_scope(absent_rag_retrieval_source_scope()):
        metadata_filter = _build_metadata_filter(RagRetrieveInput(query="hello"))
    assert metadata_filter is None


def test_validated_constructor_accepts_one_valid_source() -> None:
    scope = validated_rag_retrieval_source_scope(("source-a",))
    assert scope.is_present
    assert scope.is_invalid is False
    assert scope.allowed_source_ids == ("source-a",)


def test_validated_constructor_accepts_multiple_valid_sources() -> None:
    scope = validated_rag_retrieval_source_scope(("source-b", "source-a"))
    assert scope.is_present
    assert scope.is_invalid is False
    assert scope.allowed_source_ids == ("source-b", "source-a")


def test_validated_constructor_deduplicates_deterministically() -> None:
    scope = validated_rag_retrieval_source_scope(("source-a", "source-b", "source-a"))
    assert scope.is_invalid is False
    assert scope.allowed_source_ids == ("source-a", "source-b")


@pytest.mark.parametrize(
    ("allowed_source_ids", "expected_reason"),
    [
        ((), SOURCE_SCOPE_EMPTY),
        (("",), SOURCE_SCOPE_BLANK_ONLY),
        (("   ",), SOURCE_SCOPE_BLANK_ONLY),
        (("source-a", ""), SOURCE_SCOPE_BLANK_ONLY),
        ((123,), SOURCE_SCOPE_MALFORMED),
        (("source-a", 456), SOURCE_SCOPE_MALFORMED),
        (tuple(f"source-{index}" for index in range(21)), SOURCE_SCOPE_MALFORMED),
        (("x" * 257,), SOURCE_SCOPE_MALFORMED),
    ],
)
def test_validated_constructor_rejects_invalid_present_state(
    allowed_source_ids: tuple[object, ...],
    expected_reason: str,
) -> None:
    scope = validated_rag_retrieval_source_scope(allowed_source_ids)  # type: ignore[arg-type]
    assert scope.is_present
    assert scope.is_invalid
    assert scope.allowed_source_ids == ()
    assert scope.error_reason == expected_reason


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "presence": _ScopePresence.ABSENT,
                "allowed_source_ids": ("source-a",),
            },
            "ABSENT scope must not include allowed_source_ids",
        ),
        (
            {
                "presence": _ScopePresence.ABSENT,
                "error_reason": SOURCE_SCOPE_EMPTY,
            },
            "ABSENT scope must not include error_reason",
        ),
        (
            {
                "presence": _ScopePresence.PRESENT,
                "allowed_source_ids": ("source-a",),
                "error_reason": SOURCE_SCOPE_EMPTY,
            },
            "INVALID PRESENT scope must not include allowed_source_ids",
        ),
        (
            {
                "presence": _ScopePresence.PRESENT,
                "error_reason": "unknown_reason",
            },
            "INVALID PRESENT scope requires a recognized error_reason",
        ),
        (
            {
                "presence": _ScopePresence.PRESENT,
                "allowed_source_ids": (),
            },
            "PRESENT VALID scope requires non-empty allowed_source_ids",
        ),
        (
            {
                "presence": _ScopePresence.PRESENT,
                "allowed_source_ids": ("  ",),
            },
            "PRESENT VALID scope requires normalized non-blank source IDs",
        ),
    ],
)
def test_direct_invalid_state_construction_is_rejected(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        RagRetrievalSourceScopeState(**kwargs)  # type: ignore[arg-type]


def test_absent_state_invariant_holds() -> None:
    scope = absent_rag_retrieval_source_scope()
    assert scope.is_absent
    assert scope.allowed_source_ids == ()
    assert scope.error_reason is None
    assert scope.is_invalid is False


def test_present_valid_state_invariant_holds() -> None:
    scope = validated_rag_retrieval_source_scope(("source-a",))
    assert scope.is_present
    assert scope.allowed_source_ids == ("source-a",)
    assert scope.error_reason is None
    assert scope.is_invalid is False


def test_present_invalid_state_invariant_holds() -> None:
    scope = invalid_rag_retrieval_source_scope(SOURCE_SCOPE_MALFORMED)
    assert scope.is_present
    assert scope.allowed_source_ids == ()
    assert scope.error_reason == SOURCE_SCOPE_MALFORMED
    assert scope.is_invalid


@pytest.mark.parametrize(
    "allowed_source_ids",
    [
        (),
        ("",),
        ("   ",),
        (123,),  # type: ignore[list-item]
    ],
)
def test_perform_rag_retrieve_never_invokes_retrieval_for_validated_invalid_scope(
    allowed_source_ids: tuple[object, ...],
) -> None:
    store = FakeVectorstoreManager([])
    scope = validated_rag_retrieval_source_scope(allowed_source_ids)  # type: ignore[arg-type]
    assert scope.is_invalid
    with rag_retrieval_source_scope(scope):
        out = perform_rag_retrieve(_ctx(store), RagRetrieveInput(query="policy", tenant_id="t1"))
    assert out.used is False
    assert out.reason == scope.error_reason
    assert store.query_calls == 0


def test_parse_task_metadata_allowed_source_ids_preserves_absent_semantics() -> None:
    scope = parse_task_metadata_allowed_source_ids({})
    assert scope.is_absent
    assert scope.allowed_source_ids == ()
    assert scope.error_reason is None


@pytest.mark.parametrize(
    ("metadata_value", "expected_reason"),
    [
        ([], SOURCE_SCOPE_EMPTY),
        (["   "], SOURCE_SCOPE_BLANK_ONLY),
        ([123], SOURCE_SCOPE_MALFORMED),
    ],
)
def test_parse_task_metadata_allowed_source_ids_preserves_invalid_semantics(
    metadata_value: object,
    expected_reason: str,
) -> None:
    scope = parse_task_metadata_allowed_source_ids({"allowed_source_ids": metadata_value})
    assert scope.is_invalid
    assert scope.error_reason == expected_reason
    assert scope.allowed_source_ids == ()


def test_parse_task_metadata_allowed_source_ids_preserves_valid_semantics() -> None:
    scope = parse_task_metadata_allowed_source_ids({"allowed_source_ids": [" source-a ", "source-b"]})
    assert scope.is_present
    assert scope.is_invalid is False
    assert scope.allowed_source_ids == ("source-a", "source-b")
