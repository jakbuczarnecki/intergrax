"""Unit tests for Qdrant vector index administration plugin."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from intergrax.integrations.contracts.vector_index_administration import (
    VectorIndexCompatibilityError,
    VectorIndexIdentity,
    VectorIndexPrepareOutcome,
    VectorSearchCapability,
)
from intergrax.integrations.providers.vector_store.qdrant.config import QdrantIntegrationConfig
from intergrax.integrations.providers.vector_store.qdrant.index_administration import (
    QdrantVectorIndexAdministration,
    _dense_dimension,
    _has_sparse_channel,
    _physical_index_name,
    build_qdrant_index_spec,
)
from intergrax.integrations.providers.vector_store.qdrant.opens import _build_rag_config

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _qdrant_models_available() -> None:
    fake_distance = SimpleNamespace(COSINE="cosine", DOT="dot", EUCLID="euclidean")
    with patch(
        "intergrax.integrations.providers.vector_store.qdrant.index_administration._load_qdrant_models",
        return_value=(
            fake_distance,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            type("UnexpectedResponse", (), {}),
        ),
    ):
        yield


def _collection_info(
    *,
    dense_size: int,
    sparse_names: tuple[str, ...],
    points_count: int = 0,
) -> SimpleNamespace:
    dense = SimpleNamespace(size=dense_size)
    sparse_vectors = {name: SimpleNamespace() for name in sparse_names}
    return SimpleNamespace(
        points_count=points_count,
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors={"dense": dense},
                sparse_vectors=sparse_vectors,
            )
        ),
    )


def _identity() -> VectorIndexIdentity:
    return VectorIndexIdentity(logical_name="vpi_test", tenant_id="default")


def test_collection_dense_dimension_matches() -> None:
    info = _collection_info(dense_size=1024, sparse_names=("sparse",))
    assert _dense_dimension(info, dense_channel_name="dense") == 1024


def test_collection_sparse_channel_present() -> None:
    info = _collection_info(dense_size=1024, sparse_names=("sparse",))
    assert _has_sparse_channel(info, sparse_channel_name="sparse") is True


def test_dense_only_collection_has_no_sparse_channel() -> None:
    info = _collection_info(dense_size=1024, sparse_names=())
    assert _has_sparse_channel(info, sparse_channel_name="sparse") is False


def test_prepare_creates_missing_collection() -> None:
    client = MagicMock()
    client.get_collection.side_effect = RuntimeError("404 collection not found")
    admin = QdrantVectorIndexAdministration(
        _client=client,
        _config=QdrantIntegrationConfig(collection_name="vpi_test"),
    )
    spec = build_qdrant_index_spec(identity=_identity(), dimension=1024)
    client.get_collection.side_effect = [
        RuntimeError("404 collection not found"),
        _collection_info(dense_size=1024, sparse_names=("sparse",)),
    ]
    result = admin.prepare_index(spec)
    assert result.outcome is VectorIndexPrepareOutcome.CREATED
    client.create_collection.assert_called_once()


def test_prepare_accepts_compatible_existing_collection() -> None:
    client = MagicMock()
    client.get_collection.return_value = _collection_info(
        dense_size=1024,
        sparse_names=("sparse",),
        points_count=5,
    )
    admin = QdrantVectorIndexAdministration(
        _client=client,
        _config=QdrantIntegrationConfig(collection_name="vpi_test"),
    )
    result = admin.prepare_index(build_qdrant_index_spec(identity=_identity(), dimension=1024))
    assert result.outcome is VectorIndexPrepareOutcome.ALREADY_COMPATIBLE
    client.create_collection.assert_not_called()


def test_prepare_rejects_wrong_dimension() -> None:
    client = MagicMock()
    client.get_collection.return_value = _collection_info(
        dense_size=512,
        sparse_names=("sparse",),
        points_count=1,
    )
    admin = QdrantVectorIndexAdministration(
        _client=client,
        _config=QdrantIntegrationConfig(collection_name="vpi_test"),
    )
    with pytest.raises(VectorIndexCompatibilityError, match="dimension"):
        admin.prepare_index(build_qdrant_index_spec(identity=_identity(), dimension=1024))


def test_prepare_rejects_missing_sparse_capability() -> None:
    client = MagicMock()
    client.get_collection.return_value = _collection_info(
        dense_size=1024,
        sparse_names=(),
        points_count=0,
    )
    admin = QdrantVectorIndexAdministration(
        _client=client,
        _config=QdrantIntegrationConfig(collection_name="vpi_test"),
    )
    with pytest.raises(VectorIndexCompatibilityError, match="sparse_lexical"):
        admin.prepare_index(build_qdrant_index_spec(identity=_identity(), dimension=1024))


def test_describe_index_reports_point_count() -> None:
    client = MagicMock()
    client.get_collections.return_value = SimpleNamespace(collections=[])
    client.get_collection.return_value = _collection_info(
        dense_size=1024,
        sparse_names=("sparse",),
        points_count=42,
    )
    admin = QdrantVectorIndexAdministration(
        _client=client,
        _config=QdrantIntegrationConfig(collection_name="vpi_test"),
    )
    description = admin.describe_index(_identity())
    assert description.point_count == 42
    assert VectorSearchCapability.SPARSE_LEXICAL in description.present_capabilities


def test_probe_failure_detail_is_sanitized() -> None:
    client = MagicMock()
    client.get_collections.side_effect = RuntimeError("http://secret:key@host")
    admin = QdrantVectorIndexAdministration(
        _client=client,
        _config=QdrantIntegrationConfig(collection_name="vpi_test"),
    )
    health = admin.probe()
    assert health.healthy is False
    assert "secret" not in (health.detail or "")
    assert "RuntimeError" in (health.detail or "")


def test_close_calls_client_close() -> None:
    client = MagicMock()
    admin = QdrantVectorIndexAdministration(
        _client=client,
        _config=QdrantIntegrationConfig(collection_name="vpi_test"),
    )
    admin.close()
    client.close.assert_called_once()


def test_control_and_data_plane_physical_collection_identity_match() -> None:
    config = QdrantIntegrationConfig(
        collection_name="vpi_offers",
        tenant_id="default",
        enable_sparse_vectors=True,
    )
    identity = VectorIndexIdentity(
        logical_name=config.collection_name,
        tenant_id=config.tenant_id,
    )
    rag_config = _build_rag_config(config)
    expected = _physical_index_name(identity)
    data_plane_name = f"{rag_config.collection_name}__tenant__{rag_config.tenant_id}"
    assert data_plane_name == expected
    assert config.enable_sparse_vectors is True
    assert rag_config.enable_sparse_vectors is True
