# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.integrations.providers.vector_store.chroma.rag_store import (
    ChromaConfig,
    ChromaVectorStore,
)
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    MetadataMembershipCondition,
    VectorStoreScope,
)

pytestmark = pytest.mark.unit


def _store() -> ChromaVectorStore:
    return ChromaVectorStore(ChromaConfig(collection_name="coll", tenant_id="t1"), client=MagicMock())


def _scope() -> VectorStoreScope:
    return VectorStoreScope(tenant_id="t1", namespace="rag")


def test_chroma_membership_maps_to_in_operator() -> None:
    store = _store()
    scoped = MetadataFilter.for_scope(
        _scope(),
        MetadataFilter(
            membership=(
                MetadataMembershipCondition(
                    field="source_id",
                    allowed_values=("src-a", "src-b"),
                ),
            )
        ),
    )
    where = store._chroma_where_from_metadata(scoped)
    assert where is not None
    assert where["$and"][-1] == {"source_id": {"$in": ["src-a", "src-b"]}}


def test_chroma_equality_preserved() -> None:
    store = _store()
    scoped = MetadataFilter.for_scope(
        _scope(),
        MetadataFilter(conditions={"session_id": "s1"}),
    )
    where = store._chroma_where_from_metadata(scoped)
    assert where is not None
    assert {"$and": [{"session_id": {"$eq": "s1"}}, {"tenant_id": {"$eq": "t1"}}, {"namespace": {"$eq": "rag"}}]} == where


def test_chroma_combined_predicates_use_and() -> None:
    store = _store()
    scoped = MetadataFilter.for_scope(
        _scope(),
        MetadataFilter(
            conditions={"session_id": "s1"},
            membership=(
                MetadataMembershipCondition(field="source_id", allowed_values=("src-a",)),
            ),
        ),
    )
    where = store._chroma_where_from_metadata(scoped)
    assert where is not None
    assert "$and" in where
    terms = where["$and"]
    assert {"session_id": {"$eq": "s1"}} in terms
    assert {"source_id": {"$in": ["src-a"]}} in terms
    assert {"tenant_id": {"$eq": "t1"}} in terms


def test_chroma_query_supplies_where_before_n_results() -> None:
    store = _store()
    store._collection.query.return_value = {
        "ids": [[]],
        "distances": [[]],
        "metadatas": [[]],
        "documents": [[]],
    }
    metadata_filter = MetadataFilter(
        membership=(
            MetadataMembershipCondition(field="source_id", allowed_values=("src-a",)),
        )
    )

    store.query([0.1, 0.2], scope=_scope(), top_k=4, metadata_filter=metadata_filter)

    kwargs = store._collection.query.call_args.kwargs
    assert kwargs["n_results"] == 4
    where = kwargs["where"]
    assert any(
        term.get("source_id") == {"$in": ["src-a"]}
        for term in where.get("$and", [where])
    )


def test_chroma_empty_membership_rejected_at_contract_layer() -> None:
    with pytest.raises(Exception):
        MetadataMembershipCondition(field="source_id", allowed_values=())
