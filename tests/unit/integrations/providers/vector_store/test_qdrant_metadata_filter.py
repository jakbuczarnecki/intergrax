# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from intergrax.integrations.providers.vector_store.qdrant.rag_store import (
    QdrantConfig,
    QdrantVectorStore,
)
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    MetadataMembershipCondition,
    VectorStoreContractError,
    VectorStoreScope,
)

pytestmark = pytest.mark.unit


def _store() -> QdrantVectorStore:
    with patch(
        "intergrax.integrations.providers.vector_store.qdrant.rag_store.QdrantClient",
        return_value=MagicMock(),
    ):
        return QdrantVectorStore(QdrantConfig(collection_name="coll", tenant_id="t1"))


def _scope() -> VectorStoreScope:
    return VectorStoreScope(tenant_id="t1", namespace="rag")


def test_qdrant_membership_maps_to_match_any() -> None:
    store = _store()
    metadata_filter = MetadataFilter(
        membership=(
            MetadataMembershipCondition(
                field="source_id",
                allowed_values=("src-a", "src-b"),
            ),
        )
    )
    scoped = MetadataFilter.for_scope(_scope(), metadata_filter)
    qfilter = store._qdrant_filter_from_metadata(scoped)
    assert qfilter is not None
    must = qfilter.model_dump()["must"]
    membership = next(item for item in must if item["key"] == "source_id")
    assert membership["match"]["any"] == ["src-a", "src-b"]


def test_qdrant_equality_still_maps_to_match_value() -> None:
    store = _store()
    scoped = MetadataFilter.for_scope(
        _scope(),
        MetadataFilter(conditions={"session_id": "s1"}),
    )
    qfilter = store._qdrant_filter_from_metadata(scoped)
    assert qfilter is not None
    must = qfilter.model_dump()["must"]
    session = next(item for item in must if item["key"] == "session_id")
    assert session["match"] == {"value": "s1"}


def test_qdrant_equality_and_membership_compose_as_and() -> None:
    store = _store()
    metadata_filter = MetadataFilter(
        conditions={"session_id": "s1"},
        membership=(
            MetadataMembershipCondition(
                field="source_id",
                allowed_values=("src-a",),
            ),
        ),
    )
    scoped = MetadataFilter.for_scope(_scope(), metadata_filter)
    qfilter = store._qdrant_filter_from_metadata(scoped)
    assert qfilter is not None
    keys = {item["key"] for item in qfilter.model_dump()["must"]}
    assert {"tenant_id", "namespace", "session_id", "source_id"}.issubset(keys)


def test_qdrant_query_passes_filter_before_limit() -> None:
    store = _store()
    store._client.query_points.return_value = SimpleNamespace(points=[])
    scope = _scope()
    metadata_filter = MetadataFilter(
        membership=(
            MetadataMembershipCondition(field="source_id", allowed_values=("src-a",)),
        )
    )

    store.query([0.1, 0.2], scope=scope, top_k=3, metadata_filter=metadata_filter)

    kwargs = store._client.query_points.call_args.kwargs
    assert kwargs["limit"] == 3
    assert kwargs["query_filter"] is not None
    must = kwargs["query_filter"].model_dump()["must"]
    assert any(
        item["key"] == "source_id" and item["match"]["any"] == ["src-a"]
        for item in must
    )
    assert any(
        item["key"] == "tenant_id" and item["match"]["value"] == "t1" for item in must
    )


def test_qdrant_malformed_membership_rejected_at_contract_layer() -> None:
    with pytest.raises(Exception):
        MetadataMembershipCondition(field="source_id", allowed_values=())


def test_qdrant_membership_maps_string_and_int_scalars() -> None:
    store = _store()
    for field, allowed, expected in (
        ("source_id", ("src-a", "src-b"), ["src-a", "src-b"]),
        ("chunk_index", (1, 2), [1, 2]),
    ):
        metadata_filter = MetadataFilter(
            membership=(
                MetadataMembershipCondition(field=field, allowed_values=allowed),
            )
        )
        scoped = MetadataFilter.for_scope(_scope(), metadata_filter)
        qfilter = store._qdrant_filter_from_metadata(scoped)
        assert qfilter is not None
        must = qfilter.model_dump()["must"]
        membership = next(item for item in must if item["key"] == field)
        assert membership["match"]["any"] == expected


def test_qdrant_structured_membership_rejected_before_adapter() -> None:
    with pytest.raises(VectorStoreContractError):
        MetadataMembershipCondition(field="source_id", allowed_values=(["src-a"],))
