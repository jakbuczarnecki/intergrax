# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Qdrant vector index administration — vendor SDK ownership boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from intergrax.integrations.contracts.base import HealthStatus, IntegrationDependencyError
from intergrax.integrations.contracts.vector_index_administration import (
    DenseVectorChannelSpec,
    SparseLexicalChannelSpec,
    VectorIndexAdministration,
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

try:
    from qdrant_client.http.exceptions import UnexpectedResponse
    from qdrant_client.http.models import (
        Distance,
        SparseIndexParams,
        SparseVectorParams,
        VectorParams,
    )
except ImportError:
    UnexpectedResponse = None  # type: ignore[misc, assignment]
    Distance = None  # type: ignore[misc, assignment]
    SparseIndexParams = None  # type: ignore[misc, assignment]
    SparseVectorParams = None  # type: ignore[misc, assignment]
    VectorParams = None  # type: ignore[misc, assignment]


def _physical_index_name(identity: VectorIndexIdentity) -> str:
    return f"{identity.logical_name}__tenant__{identity.tenant_id}"


def _is_index_not_found(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if UnexpectedResponse is not None and isinstance(current, UnexpectedResponse):
            if current.status_code == 404:
                return True
        current = current.__cause__
    return "404" in str(exc)


def _dense_dimension(
    collection_info: object,
    *,
    dense_channel_name: str,
) -> int | None:
    try:
        vectors = collection_info.config.params.vectors  # type: ignore[attr-defined]
    except AttributeError:
        return None
    if vectors is None:
        return None
    if isinstance(vectors, Mapping):
        dense = vectors.get(dense_channel_name)
        if dense is not None:
            return int(dense.size)
        if len(vectors) == 1:
            only = next(iter(vectors.values()))
            return int(only.size)
        return None
    return int(vectors.size)


def _has_sparse_channel(
    collection_info: object,
    *,
    sparse_channel_name: str,
) -> bool:
    try:
        sparse_vectors = collection_info.config.params.sparse_vectors  # type: ignore[attr-defined]
    except AttributeError:
        return False
    if sparse_vectors is None:
        return False
    if isinstance(sparse_vectors, Mapping):
        return sparse_channel_name in sparse_vectors
    return False


def _point_count(collection_info: object) -> int:
    try:
        return int(collection_info.points_count)  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError):
        return 0


def _present_capabilities(
    collection_info: object,
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


def _distance_for_metric(metric: str) -> object:
    if Distance is None:
        raise IntegrationDependencyError("qdrant-client is not installed")
    mapping = {
        "cosine": Distance.COSINE,
        "dot": Distance.DOT,
        "euclidean": Distance.EUCLID,
    }
    return mapping.get(metric, Distance.COSINE)


def _description_from_collection(
  identity: VectorIndexIdentity,
  collection_info: object | None,
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

    _client: object
    _config: QdrantIntegrationConfig
    _default_dense_channel: str = "dense"
    _default_sparse_channel: str = "sparse"

    def probe(self) -> HealthStatus:
        try:
            self._client.get_collections()
            return HealthStatus(slug="qdrant", healthy=True, detail="get_collections succeeded")
        except Exception as exc:
            return HealthStatus(slug="qdrant", healthy=False, detail=str(exc))

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

    def _get_collection_info(self, identity: VectorIndexIdentity) -> object | None:
        physical_name = _physical_index_name(identity)
        try:
            return self._client.get_collection(physical_name)
        except Exception as exc:
            if _is_index_not_found(exc):
                return None
            raise IntegrationDependencyError("qdrant get_collection failed") from exc

    def _create_collection(self, spec: VectorIndexSpec) -> None:
        if VectorParams is None or Distance is None:
            raise IntegrationDependencyError("qdrant-client is not installed")
        physical_name = _physical_index_name(spec.identity)
        dist = _distance_for_metric(spec.dense.metric)
        sparse_required = VectorSearchCapability.SPARSE_LEXICAL in spec.required_capabilities
        try:
            if sparse_required:
                if SparseVectorParams is None or SparseIndexParams is None:
                    raise IntegrationDependencyError("qdrant sparse vectors are unavailable")
                self._client.create_collection(
                    collection_name=physical_name,
                    vectors_config={
                        spec.dense.channel_name: VectorParams(
                            size=spec.dense.dimension,
                            distance=dist,
                        ),
                    },
                    sparse_vectors_config={
                        (spec.sparse_lexical.channel_name if spec.sparse_lexical else self._default_sparse_channel): SparseVectorParams(
                            index=SparseIndexParams(),
                        ),
                    },
                )
            else:
                self._client.create_collection(
                    collection_name=physical_name,
                    vectors_config=VectorParams(
                        size=spec.dense.dimension,
                        distance=dist,
                    ),
                )
        except Exception as exc:
            raise IntegrationDependencyError("qdrant create_collection failed") from exc


def build_qdrant_index_spec(
    *,
    identity: VectorIndexIdentity,
    dimension: int,
    metric: str = "cosine",
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
            metric=metric,  # type: ignore[arg-type]
        ),
        required_capabilities=frozenset(required),
        sparse_lexical=sparse_spec,
    )
