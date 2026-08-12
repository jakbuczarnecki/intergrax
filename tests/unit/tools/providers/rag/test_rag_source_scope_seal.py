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
    absent_rag_retrieval_source_scope,
    invalid_rag_retrieval_source_scope,
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
