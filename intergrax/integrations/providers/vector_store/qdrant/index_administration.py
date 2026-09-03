# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Qdrant vector index administration — vendor SDK ownership boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from intergrax.integrations.contracts.base import (
    HealthStatus,
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.contracts.vector_index_administration import (
    DenseVectorChannelSpec,
    SparseLexicalChannelSpec,
    VectorIndexCompatibilityError,
    VectorIndexDescription,
    VectorIndexIdentity,
    VectorIndexPrepareOutcome,
    VectorIndexPrepareResult,
    VectorIndexSpec,
    VectorSearchCapability,
    validate_spec_against_description,
)
from intergrax.integrations.providers.vector_store.qdrant.config import QdrantIntegrationConfig
from intergrax.rag.vectorstore.config.vector_config import Metric

if TYPE_CHECKING:
    from qdrant_client.http.exceptions import UnexpectedResponse
    from qdrant_client.http.models import (
        CollectionInfo,
        CollectionsResponse,
        Distance,
        SparseIndexParams,
        SparseVectorParams,
        VectorParams,
    )


@dataclass(frozen=True, slots=True)
class _QdrantModelTypes:
    distance: type[Distance]
    vector_params: type[VectorParams]
    sparse_vector_params: type[SparseVectorParams]
    sparse_index_params: type[SparseIndexParams]
    unexpected_response: type[UnexpectedResponse]


@runtime_checkable
class QdrantControlPlaneClient(Protocol):
    """Narrow Qdrant client surface used by the control-plane plugin."""

    def get_collections(self) -> CollectionsResponse: ...

    def get_collection(self, collection_name: str) -> CollectionInfo: ...

    def create_collection(
        self,
        *,
        collection_name: str,
        vectors_config: VectorParams | dict[str, VectorParams],
        sparse_vectors_config: dict[str, SparseVectorParams] | None = ...,
    ) -> bool: ...

    def close(self) -> None: ...


def _load_qdrant_models() -> _QdrantModelTypes:
    try:
        from qdrant_client.http.exceptions import UnexpectedResponse
        from qdrant_client.http.models import (
            Distance,
            SparseIndexParams,
            SparseVectorParams,
            VectorParams,
        )
    except ImportError as exc:
        raise IntegrationDependencyError(
            "qdrant-client is not installed; install Intergrax-ai[vector-qdrant]"
        ) from exc
    return _QdrantModelTypes(
        distance=Distance,
        vector_params=VectorParams,
        sparse_vector_params=SparseVectorParams,
        sparse_index_params=SparseIndexParams,
        unexpected_response=UnexpectedResponse,
    )


def _physical_index_name(identity: VectorIndexIdentity) -> str:
    return f"{identity.logical_name}__tenant__{identity.tenant_id}"


def _is_index_not_found(
    exc: BaseException,
    *,
    unexpected_response_type: type[UnexpectedResponse],
) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, unexpected_response_type) and current.status_code == 404:
            return True
        current = current.__cause__
    return False


def _dense_dimension(
    collection_info: CollectionInfo,
    *,
    dense_channel_name: str,
) -> int | None:
    vectors = collection_info.config.params.vectors
    if vectors is None:
        return None
    if isinstance(vectors, dict):
        dense = vectors.get(dense_channel_name)
        if dense is not None:
            return int(dense.size)
        if len(vectors) == 1:
            only = next(iter(vectors.values()))
            return int(only.size)
        return None
    return int(vectors.size)


def _has_sparse_channel(
    collection_info: CollectionInfo,
    *,
    sparse_channel_name: str,
) -> bool:
    sparse_vectors = collection_info.config.params.sparse_vectors
    if sparse_vectors is None:
        return False
    return sparse_channel_name in sparse_vectors


def _point_count(collection_info: CollectionInfo) -> int:
    count = collection_info.points_count
    if count is None or count < 0:
        return 0
    return int(count)


def _present_capabilities(
    collection_info: CollectionInfo,
    *,
    dense_channel_name: str,
    sparse_channel_name: str,
) -> frozenset[VectorSearchCapability]:
    capabilities: set[VectorSearchCapability] = set()
    if _dense_dimension(collection_info, dense_channel_name=dense_channel_name) is not None:
        capabilities.add(VectorSearchCapability.DENSE)
    if _has_sparse_channel(collection_info, sparse_channel_name=sparse_channel_name):
        capabilities.add(VectorSearchCapability.SPARSE_LEXICAL)
    return frozenset(capabilities)


def _distance_for_metric(metric: Metric, models: _QdrantModelTypes) -> Distance:
    match metric:
        case "cosine":
            return models.distance.COSINE
        case "dot":
            return models.distance.DOT
        case "euclidean":
            return models.distance.EUCLID
        case _:
            raise IntegrationConfigurationError(f"unsupported dense metric: {metric!r}")


def _sanitize_probe_detail(exc: BaseException) -> str:
    return f"qdrant probe failed: {type(exc).__name__}"


def _description_from_collection(
    identity: VectorIndexIdentity,
    collection_info: CollectionInfo | None,
    *,
    dense_channel_name: str,
    sparse_channel_name: str,
    reachable: bool,
) -> VectorIndexDescription:
    if collection_info is None:
        return VectorIndexDescription(
            identity=identity,
            exists=False,
            reachable=reachable,
            point_count=0,
            dense_dimension=None,
            present_capabilities=frozenset(),
            dense_channel_name=None,
            sparse_lexical_channel_name=None,
        )
    dense_dimension = _dense_dimension(collection_info, dense_channel_name=dense_channel_name)
    sparse_present = _has_sparse_channel(
        collection_info,
        sparse_channel_name=sparse_channel_name,
    )
    return VectorIndexDescription(
        identity=identity,
        exists=True,
        reachable=reachable,
        point_count=_point_count(collection_info),
        dense_dimension=dense_dimension,
        present_capabilities=_present_capabilities(
            collection_info,
            dense_channel_name=dense_channel_name,
            sparse_channel_name=sparse_channel_name,
        ),
        dense_channel_name=dense_channel_name if dense_dimension is not None else None,
        sparse_lexical_channel_name=sparse_channel_name if sparse_present else None,
    )


@dataclass(slots=True)
class QdrantVectorIndexAdministration:
    """Qdrant implementation of ``VectorIndexAdministration``."""

    _client: QdrantControlPlaneClient
    _config: QdrantIntegrationConfig
    _default_dense_channel: str = "dense"
    _default_sparse_channel: str = "sparse"

    def probe(self) -> HealthStatus:
        try:
            self._client.get_collections()
            return HealthStatus(slug="qdrant", healthy=True, detail="get_collections succeeded")
        except Exception as exc:
            return HealthStatus(
                slug="qdrant",
                healthy=False,
                detail=_sanitize_probe_detail(exc),
            )

    def describe_index(self, identity: VectorIndexIdentity) -> VectorIndexDescription:
        reachable = self.probe().healthy
        collection_info = self._get_collection_info(identity)
        dense_channel = self._default_dense_channel
        sparse_channel = self._default_sparse_channel
        return _description_from_collection(
            identity,
            collection_info,
            dense_channel_name=dense_channel,
            sparse_channel_name=sparse_channel,
            reachable=reachable,
        )

    def prepare_index(self, spec: VectorIndexSpec) -> VectorIndexPrepareResult:
        dense_channel = spec.dense.channel_name
        sparse_channel = (
            spec.sparse_lexical.channel_name
            if spec.sparse_lexical is not None
            else self._default_sparse_channel
        )
        existing = self._get_collection_info(spec.identity)
        if existing is None:
            self._create_collection(spec)
            description = _description_from_collection(
                spec.identity,
                self._get_collection_info(spec.identity),
                dense_channel_name=dense_channel,
                sparse_channel_name=sparse_channel,
                reachable=True,
            )
            return VectorIndexPrepareResult(
                outcome=VectorIndexPrepareOutcome.CREATED,
                description=description,
            )

        description = _description_from_collection(
            spec.identity,
            existing,
            dense_channel_name=dense_channel,
            sparse_channel_name=sparse_channel,
            reachable=True,
        )
        validate_spec_against_description(spec, description)
        return VectorIndexPrepareResult(
            outcome=VectorIndexPrepareOutcome.ALREADY_COMPATIBLE,
            description=description,
        )

    def close(self) -> None:
        self._client.close()

    def _get_collection_info(self, identity: VectorIndexIdentity) -> CollectionInfo | None:
        physical_name = _physical_index_name(identity)
        models = _load_qdrant_models()
        try:
            return self._client.get_collection(physical_name)
        except models.unexpected_response as exc:
            if _is_index_not_found(exc, unexpected_response_type=models.unexpected_response):
                return None
            raise IntegrationConfigurationError("qdrant get_collection failed") from exc
        except Exception as exc:
            raise IntegrationDependencyError("qdrant get_collection failed") from exc

    def _create_collection(self, spec: VectorIndexSpec) -> None:
        models = _load_qdrant_models()
        physical_name = _physical_index_name(spec.identity)
        dist = _distance_for_metric(spec.dense.metric, models)
        sparse_required = VectorSearchCapability.SPARSE_LEXICAL in spec.required_capabilities
        try:
            if sparse_required:
                sparse_channel = (
                    spec.sparse_lexical.channel_name
                    if spec.sparse_lexical is not None
                    else self._default_sparse_channel
                )
                self._client.create_collection(
                    collection_name=physical_name,
                    vectors_config={
                        spec.dense.channel_name: models.vector_params(
                            size=spec.dense.dimension,
                            distance=dist,
                        ),
                    },
                    sparse_vectors_config={
                        sparse_channel: models.sparse_vector_params(
                            index=models.sparse_index_params(),
                        ),
                    },
                )
            else:
                self._client.create_collection(
                    collection_name=physical_name,
                    vectors_config=models.vector_params(
                        size=spec.dense.dimension,
                        distance=dist,
                    ),
                )
        except IntegrationDependencyError:
            raise
        except IntegrationConfigurationError:
            raise
        except Exception as exc:
            raise IntegrationConfigurationError("qdrant create_collection failed") from exc


def build_qdrant_index_spec(
    *,
    identity: VectorIndexIdentity,
    dimension: int,
    metric: Metric = "cosine",
    enable_sparse_lexical: bool = True,
    dense_channel_name: str = "dense",
    sparse_channel_name: str = "sparse",
) -> VectorIndexSpec:
    required: set[VectorSearchCapability] = {VectorSearchCapability.DENSE}
    sparse_spec: SparseLexicalChannelSpec | None = None
    if enable_sparse_lexical:
        required.add(VectorSearchCapability.SPARSE_LEXICAL)
        sparse_spec = SparseLexicalChannelSpec(channel_name=sparse_channel_name)
    return VectorIndexSpec(
        identity=identity,
        dense=DenseVectorChannelSpec(
            channel_name=dense_channel_name,
            dimension=dimension,
            metric=metric,
        ),
        required_capabilities=frozenset(required),
        sparse_lexical=sparse_spec,
    )
