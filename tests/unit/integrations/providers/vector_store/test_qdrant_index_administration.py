"""Unit tests for Qdrant vector index administration plugin."""

from __future__ import annotations

import builtins
from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.contracts.vector_index_administration import (
    VectorIndexCompatibilityError,
    VectorIndexIdentity,
    VectorIndexPrepareOutcome,
    VectorSearchCapability,
)
from intergrax.integrations.providers.vector_store.qdrant.config import QdrantIntegrationConfig
from intergrax.integrations.providers.vector_store.qdrant.index_administration import (
    QdrantVectorIndexAdministration,
    _QdrantModelTypes,
    _dense_dimension,
    _distance_for_metric,
    _has_sparse_channel,
    _is_index_not_found,
    _physical_index_name,
    build_qdrant_index_spec,
)
from intergrax.integrations.providers.vector_store.qdrant.opens import (
    _build_rag_config,
    open_qdrant_vector_index_administration,
)

pytestmark = pytest.mark.unit


class _FakeDistance(Enum):
    COSINE = "cosine"
    DOT = "dot"
    EUCLID = "euclidean"


class FakeUnexpectedResponse(Exception):
    def __init__(self, status_code: int, message: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class _FakeVectorParams:
    size: int
    distance: _FakeDistance


@dataclass(frozen=True, slots=True)
class _FakeSparseVectorParams:
    index: object


@dataclass(frozen=True, slots=True)
class _FakeCollectionParams:
    vectors: dict[str, _FakeVectorParams] | _FakeVectorParams | None
    sparse_vectors: dict[str, _FakeSparseVectorParams] | None


@dataclass(frozen=True, slots=True)
class _FakeCollectionConfig:
    params: _FakeCollectionParams


@dataclass(frozen=True, slots=True)
class _FakeCollectionInfo:
    points_count: int
    config: _FakeCollectionConfig


@dataclass(frozen=True, slots=True)
class _FakeCollectionsResponse:
    collections: tuple[object, ...]


class _FakeControlPlaneClient:
    def __init__(
        self,
        *,
        collection_info: _FakeCollectionInfo | None = None,
        get_collection_side_effect: BaseException | None = None,
    ) -> None:
        self._collection_info = collection_info
        self._get_collection_side_effect = get_collection_side_effect
        self.get_collection_calls: list[str] = []
        self.create_collection_calls: list[dict[str, object]] = []
        self.closed = False

    def get_collections(self) -> _FakeCollectionsResponse:
        return _FakeCollectionsResponse(collections=())

    def get_collection(self, collection_name: str) -> _FakeCollectionInfo:
        self.get_collection_calls.append(collection_name)
        if self._get_collection_side_effect is not None:
            raise self._get_collection_side_effect
        if self._collection_info is None:
            raise FakeUnexpectedResponse(404, "collection not found")
        return self._collection_info

    def create_collection(
        self,
        *,
        collection_name: str,
        vectors_config: _FakeVectorParams | dict[str, _FakeVectorParams],
        sparse_vectors_config: dict[str, _FakeSparseVectorParams] | None = None,
    ) -> bool:
        self.create_collection_calls.append(
            {
                "collection_name": collection_name,
                "vectors_config": vectors_config,
                "sparse_vectors_config": sparse_vectors_config,
            }
        )
        return True

    def close(self) -> None:
        self.closed = True


def _fake_qdrant_models() -> _QdrantModelTypes:
    return _QdrantModelTypes(
        distance=_FakeDistance,
        vector_params=_FakeVectorParams,
        sparse_vector_params=_FakeSparseVectorParams,
        sparse_index_params=MagicMock,
        unexpected_response=FakeUnexpectedResponse,
    )


@pytest.fixture(autouse=True)
def _qdrant_models_available(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    if request.node.name == "test_load_qdrant_models_raises_integration_dependency_error":
        yield
        return
    with patch(
        "intergrax.integrations.providers.vector_store.qdrant.index_administration._load_qdrant_models",
        return_value=_fake_qdrant_models(),
    ):
        yield


def _collection_info(
    *,
    dense_size: int,
    sparse_names: tuple[str, ...],
    points_count: int = 0,
) -> _FakeCollectionInfo:
    dense = _FakeVectorParams(size=dense_size, distance=_FakeDistance.COSINE)
    sparse_vectors = {name: _FakeSparseVectorParams(index=object()) for name in sparse_names}
    return _FakeCollectionInfo(
        points_count=points_count,
        config=_FakeCollectionConfig(
            params=_FakeCollectionParams(
                vectors={"dense": dense},
                sparse_vectors=sparse_vectors or None,
            )
        ),
    )


def _identity() -> VectorIndexIdentity:
    return VectorIndexIdentity(logical_name="vpi_test", tenant_id="default")


def _admin(
    client: _FakeControlPlaneClient,
    *,
    collection_name: str = "vpi_test",
) -> QdrantVectorIndexAdministration:
    return QdrantVectorIndexAdministration(
        _client=client,
        _config=QdrantIntegrationConfig(collection_name=collection_name),
    )


def test_collection_dense_dimension_matches() -> None:
    info = _collection_info(dense_size=1024, sparse_names=("sparse",))
    assert _dense_dimension(info, dense_channel_name="dense") == 1024


def test_collection_sparse_channel_present() -> None:
    info = _collection_info(dense_size=1024, sparse_names=("sparse",))
    assert _has_sparse_channel(info, sparse_channel_name="sparse") is True


def test_dense_only_collection_has_no_sparse_channel() -> None:
    info = _collection_info(dense_size=1024, sparse_names=())
    assert _has_sparse_channel(info, sparse_channel_name="sparse") is False


def test_load_qdrant_models_raises_integration_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def missing_qdrant(name: str, *args: object, **kwargs: object) -> object:
        if name == "qdrant_client" or name.startswith("qdrant_client."):
            raise ModuleNotFoundError("No module named 'qdrant_client'", name="qdrant_client")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_qdrant)
    from intergrax.integrations.providers.vector_store.qdrant import index_administration

    with pytest.raises(IntegrationDependencyError, match="vector-qdrant"):
        index_administration._load_qdrant_models()


def test_probe_success() -> None:
    client = _FakeControlPlaneClient(collection_info=_collection_info(dense_size=8, sparse_names=()))
    health = _admin(client).probe()
    assert health.healthy is True
    assert health.detail == "get_collections succeeded"


def test_probe_failure_detail_is_sanitized() -> None:
    client = _FakeControlPlaneClient(collection_info=_collection_info(dense_size=8, sparse_names=()))

    def _raise_secret() -> _FakeCollectionsResponse:
        raise RuntimeError("http://secret:key@host")

    client.get_collections = _raise_secret  # type: ignore[method-assign]
    health = _admin(client).probe()
    assert health.healthy is False
    assert "secret" not in (health.detail or "")
    assert "RuntimeError" in (health.detail or "")


def test_describe_missing_collection() -> None:
    client = _FakeControlPlaneClient(
        get_collection_side_effect=FakeUnexpectedResponse(404, "missing"),
    )
    description = _admin(client).describe_index(_identity())
    assert description.exists is False
    assert description.point_count == 0
    assert description.present_capabilities == frozenset()


def test_describe_dense_collection() -> None:
    client = _FakeControlPlaneClient(
        collection_info=_collection_info(dense_size=1024, sparse_names=()),
    )
    description = _admin(client).describe_index(_identity())
    assert description.exists is True
    assert description.dense_dimension == 1024
    assert description.present_capabilities == frozenset({VectorSearchCapability.DENSE})
    assert description.sparse_lexical_channel_name is None


def test_describe_dense_and_sparse_collection() -> None:
    client = _FakeControlPlaneClient(
        collection_info=_collection_info(dense_size=1024, sparse_names=("sparse",)),
    )
    description = _admin(client).describe_index(_identity())
    assert VectorSearchCapability.DENSE in description.present_capabilities
    assert VectorSearchCapability.SPARSE_LEXICAL in description.present_capabilities
    assert description.sparse_lexical_channel_name == "sparse"


def test_describe_index_reports_point_count() -> None:
    client = _FakeControlPlaneClient(
        collection_info=_collection_info(
            dense_size=1024,
            sparse_names=("sparse",),
            points_count=42,
        ),
    )
    description = _admin(client).describe_index(_identity())
    assert description.point_count == 42


def test_prepare_creates_missing_collection() -> None:
    created_info = _collection_info(dense_size=1024, sparse_names=("sparse",))
    responses: list[BaseException | _FakeCollectionInfo] = [
        FakeUnexpectedResponse(404, "missing"),
        created_info,
        created_info,
    ]

    class _SequentialClient(_FakeControlPlaneClient):
        def get_collection(self, collection_name: str) -> _FakeCollectionInfo:
            self.get_collection_calls.append(collection_name)
            next_item = responses.pop(0)
            if isinstance(next_item, BaseException):
                raise next_item
            return next_item

    client = _SequentialClient()
    admin = _admin(client)
    spec = build_qdrant_index_spec(identity=_identity(), dimension=1024)
    result = admin.prepare_index(spec)
    assert result.outcome is VectorIndexPrepareOutcome.CREATED
    assert len(client.create_collection_calls) == 1


def test_prepare_accepts_compatible_existing_collection() -> None:
    client = _FakeControlPlaneClient(
        collection_info=_collection_info(
            dense_size=1024,
            sparse_names=("sparse",),
            points_count=5,
        ),
    )
    result = _admin(client).prepare_index(build_qdrant_index_spec(identity=_identity(), dimension=1024))
    assert result.outcome is VectorIndexPrepareOutcome.ALREADY_COMPATIBLE
    assert client.create_collection_calls == []


def test_prepare_rejects_wrong_dimension() -> None:
    client = _FakeControlPlaneClient(
        collection_info=_collection_info(
            dense_size=512,
            sparse_names=("sparse",),
            points_count=1,
        ),
    )
    with pytest.raises(VectorIndexCompatibilityError, match="dimension"):
        _admin(client).prepare_index(build_qdrant_index_spec(identity=_identity(), dimension=1024))


def test_prepare_rejects_missing_sparse_capability() -> None:
    client = _FakeControlPlaneClient(
        collection_info=_collection_info(dense_size=1024, sparse_names=(), points_count=0),
    )
    with pytest.raises(VectorIndexCompatibilityError, match="sparse_lexical"):
        _admin(client).prepare_index(build_qdrant_index_spec(identity=_identity(), dimension=1024))


def test_vendor_404_translated_to_not_exists() -> None:
    client = _FakeControlPlaneClient(
        get_collection_side_effect=FakeUnexpectedResponse(404, "not found"),
    )
    admin = _admin(client)
    assert admin._get_collection_info(_identity()) is None


def test_non_404_vendor_failure_translated_to_configuration_error() -> None:
    client = _FakeControlPlaneClient(
        get_collection_side_effect=FakeUnexpectedResponse(503, "unavailable"),
    )
    with pytest.raises(IntegrationConfigurationError, match="get_collection failed"):
        _admin(client).describe_index(_identity())


def test_non_vendor_failure_translated_to_dependency_error() -> None:
    client = _FakeControlPlaneClient(
        get_collection_side_effect=RuntimeError("connection reset"),
    )
    with pytest.raises(IntegrationDependencyError, match="get_collection failed"):
        _admin(client).describe_index(_identity())


def test_close_calls_client_close() -> None:
    client = _FakeControlPlaneClient(collection_info=_collection_info(dense_size=8, sparse_names=()))
    _admin(client).close()
    assert client.closed is True


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("cosine", _FakeDistance.COSINE),
        ("dot", _FakeDistance.DOT),
        ("euclidean", _FakeDistance.EUCLID),
    ],
)
def test_metric_mapping(metric: str, expected: _FakeDistance) -> None:
    assert _distance_for_metric(metric, _fake_qdrant_models()) == expected


def test_metric_mapping_rejects_unknown_metric() -> None:
    with pytest.raises(IntegrationConfigurationError, match="unsupported dense metric"):
        _distance_for_metric("manhattan", _fake_qdrant_models())  # type: ignore[arg-type]


def test_is_index_not_found_uses_typed_status_code_only() -> None:
    assert _is_index_not_found(
        FakeUnexpectedResponse(404),
        unexpected_response_type=FakeUnexpectedResponse,
    )
    assert not _is_index_not_found(
        RuntimeError("404 collection not found"),
        unexpected_response_type=FakeUnexpectedResponse,
    )


def test_physical_index_name() -> None:
    identity = VectorIndexIdentity(logical_name="vpi_offers", tenant_id="default")
    assert _physical_index_name(identity) == "vpi_offers__tenant__default"


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


def test_open_qdrant_vector_index_administration_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def missing_qdrant(name: str, *args: object, **kwargs: object) -> object:
        if name == "qdrant_client":
            raise ModuleNotFoundError("No module named 'qdrant_client'", name="qdrant_client")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_qdrant)
    config = QdrantIntegrationConfig(collection_name="vpi_test")
    with pytest.raises(IntegrationConfigurationError, match="vector-qdrant"):
        open_qdrant_vector_index_administration(config)


def test_generic_vector_index_contract_imports_without_qdrant_client() -> None:
    import importlib

    module = importlib.import_module("intergrax.integrations.contracts.vector_index_administration")
    source = module.__file__
    assert source is not None
    text = open(source, encoding="utf-8").read()
    assert "qdrant_client" not in text
    assert "qdrant" not in text.casefold()
