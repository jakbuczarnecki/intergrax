"""Qdrant-backed resolver for vector index runtime identity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from intergrax.integrations.contracts.base import IntegrationDependencyError
from intergrax.integrations.contracts.vector_index_administration import (
    VectorIndexAdministration,
    VectorIndexIdentity,
)
from intergrax.integrations.providers.vector_store.qdrant.config import QdrantIntegrationConfig
from intergrax.integrations.providers.vector_store.qdrant.opens import (
    _build_qdrant_client,
)
from intergrax.integrations.providers.vector_store.qdrant.rag_store import (
    _normalize_point_id,
)
from intergrax.rag.vectorstore.config.vector_config import Metric

from platform_proofs.scenarios.verified_product_identification.integrations.search_store.vector_index_compatibility import (
    VectorIndexIdentityResolver,
)
from platform_proofs.scenarios.verified_product_identification.integrations.search_store.vector_index_metadata import (
    INDEX_METADATA_LOGICAL_POINT_ID,
    VectorIndexPersistedMetadata,
)
from platform_proofs.scenarios.verified_product_identification.integrations.search_store.vector_index_runtime_identity import (
    ExpectedVectorIndexRuntimeIdentity,
    ResolvedVectorIndexRuntimeIdentity,
    VectorIndexIdentityResolution,
    VectorIndexIdentityResolutionStatus,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.errors import (
    QdrantBootstrapOperationError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.payload import (
    EMBEDDING_DIMENSION_PAYLOAD_KEY,
    EMBEDDING_MODEL_PAYLOAD_KEY,
    EMBEDDING_PROVIDER_PAYLOAD_KEY,
    EMBEDDING_REVISION_PAYLOAD_KEY,
    payload_from_provider_dict,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.target_mapping import (
    physical_collection_name,
)

type QdrantProviderDistance = str
type QdrantProviderPayloadInput = Mapping[str, str | int] | None


class QdrantVectorParamsView(Protocol):
    size: int
    distance: QdrantProviderDistance


class QdrantCollectionParamsView(Protocol):
    vectors: QdrantVectorParamsView | dict[str, QdrantVectorParamsView] | None


class QdrantCollectionConfigView(Protocol):
    params: QdrantCollectionParamsView


class QdrantCollectionInfoView(Protocol):
    config: QdrantCollectionConfigView


class QdrantProviderPoint(Protocol):
    id: str | int
    payload: QdrantProviderPayloadInput


class QdrantIndexProbeClient(Protocol):
    def get_collection(self, collection_name: str) -> QdrantCollectionInfoView: ...

    def retrieve(
        self,
        collection_name: str,
        ids: Sequence[str | int],
        *,
        with_payload: bool,
        with_vectors: bool,
    ) -> Sequence[QdrantProviderPoint]: ...

    def scroll(
        self,
        collection_name: str,
        *,
        limit: int,
        with_payload: bool,
        with_vectors: bool,
    ) -> tuple[Sequence[QdrantProviderPoint], str | int | None]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _CollectionVectorShape:
    dimension: int
    distance: str


def _physical_index_name(identity: VectorIndexIdentity) -> str:
    return physical_collection_name(identity.logical_name, identity.tenant_id)


def _distance_label(distance: QdrantProviderDistance) -> str:
    return str(distance)


def _metric_from_distance_label(distance: str) -> Metric | None:
    normalized = distance.strip().lower()
    if normalized in {"cosine", "distance.cosine"}:
        return "cosine"
    if normalized in {"dot", "distance.dot"}:
        return "dot"
    if normalized in {"euclid", "euclidean", "distance.euclid"}:
        return "euclidean"
    return None


def _collection_vector_shape(collection_info: QdrantCollectionInfoView) -> _CollectionVectorShape:
    vectors = collection_info.config.params.vectors
    if vectors is None:
        raise ValueError("collection has no dense vector config")
    if isinstance(vectors, dict):
        if len(vectors) != 1:
            raise ValueError("collection dense vector channel is ambiguous")
        dense = next(iter(vectors.values()))
    else:
        dense = vectors
    return _CollectionVectorShape(
        dimension=int(dense.size),
        distance=_distance_label(dense.distance),
    )


def _extract_provider_payload(raw_payload: QdrantProviderPayloadInput) -> dict[str, str | int]:
    if raw_payload is None:
        raise ValueError("stored payload missing")
    converted: dict[str, str | int] = {}
    for key, value in raw_payload.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, str) or isinstance(value, int):
            converted[key] = value
    return converted


@dataclass(slots=True)
class QdrantVectorIndexIdentityResolver:
    """Resolve Qdrant index identity via administration and typed payload probes."""

    _index_admin: VectorIndexAdministration
    _probe_client: QdrantIndexProbeClient

    @classmethod
    def from_qdrant_config(
        cls,
        config: QdrantIntegrationConfig,
        *,
        index_admin: VectorIndexAdministration | None = None,
    ) -> QdrantVectorIndexIdentityResolver:
        if index_admin is None:
            from intergrax.integrations.providers.vector_store.qdrant.opens import (
                open_qdrant_vector_index_administration,
            )

            resolved_admin = open_qdrant_vector_index_administration(config)
        else:
            resolved_admin = index_admin
        return cls(
            _index_admin=resolved_admin,
            _probe_client=_build_qdrant_client(config),
        )

    def close(self) -> None:
        self._index_admin.close()
        self._probe_client.close()

    def resolve(
        self,
        expected: ExpectedVectorIndexRuntimeIdentity,
    ) -> VectorIndexIdentityResolution:
        try:
            description = self._index_admin.describe_index(expected.target)
        except (OSError, ConnectionError, TimeoutError, IntegrationDependencyError):
            return VectorIndexIdentityResolution(
                status=VectorIndexIdentityResolutionStatus.PROVIDER_UNAVAILABLE,
                identity=None,
            )
        if not description.reachable:
            return VectorIndexIdentityResolution(
                status=VectorIndexIdentityResolutionStatus.PROVIDER_UNAVAILABLE,
                identity=None,
            )
        if not description.exists:
            return VectorIndexIdentityResolution(
                status=VectorIndexIdentityResolutionStatus.INDEX_MISSING,
                identity=None,
            )
        try:
            collection_info = self._probe_client.get_collection(_physical_index_name(expected.target))
            shape = _collection_vector_shape(collection_info)
        except (OSError, ConnectionError, TimeoutError, ValueError):
            return VectorIndexIdentityResolution(
                status=VectorIndexIdentityResolutionStatus.METADATA_UNAVAILABLE,
                identity=None,
            )
        metric = _metric_from_distance_label(shape.distance)
        metadata_payload = self._read_index_metadata_payload(expected.target)
        if metadata_payload is not None:
            try:
                metadata = VectorIndexPersistedMetadata.from_provider_payload(
                    metadata_payload,
                    embedding_provider_key=EMBEDDING_PROVIDER_PAYLOAD_KEY,
                    embedding_model_key=EMBEDDING_MODEL_PAYLOAD_KEY,
                    embedding_revision_key=EMBEDDING_REVISION_PAYLOAD_KEY,
                    embedding_dimension_key=EMBEDDING_DIMENSION_PAYLOAD_KEY,
                )
            except ValueError:
                return VectorIndexIdentityResolution(
                    status=VectorIndexIdentityResolutionStatus.METADATA_UNAVAILABLE,
                    identity=None,
                )
            return VectorIndexIdentityResolution(
                status=VectorIndexIdentityResolutionStatus.RESOLVED,
                identity=ResolvedVectorIndexRuntimeIdentity(
                    target=expected.target,
                    exists=True,
                    reachable=True,
                    provider=metadata.provider,
                    model=metadata.model,
                    revision=metadata.revision,
                    dimension=shape.dimension,
                    metric=metric,
                    content_identity=metadata.content_identity,
                ),
            )
        embedding_payload = self._read_embedding_probe_payload(expected.target)
        if embedding_payload is None:
            return VectorIndexIdentityResolution(
                status=VectorIndexIdentityResolutionStatus.METADATA_UNAVAILABLE,
                identity=None,
            )
        try:
            payload = payload_from_provider_dict(embedding_payload)
        except QdrantBootstrapOperationError:
            return VectorIndexIdentityResolution(
                status=VectorIndexIdentityResolutionStatus.METADATA_UNAVAILABLE,
                identity=None,
            )
        if payload.embedding_revision is None:
            return VectorIndexIdentityResolution(
                status=VectorIndexIdentityResolutionStatus.METADATA_UNAVAILABLE,
                identity=None,
            )
        return VectorIndexIdentityResolution(
            status=VectorIndexIdentityResolutionStatus.RESOLVED,
            identity=ResolvedVectorIndexRuntimeIdentity(
                target=expected.target,
                exists=True,
                reachable=True,
                provider=payload.embedding_provider,
                model=payload.embedding_model,
                revision=payload.embedding_revision,
                dimension=shape.dimension,
                metric=metric,
                content_identity=None,
            ),
        )

    def _read_index_metadata_payload(
        self,
        target: VectorIndexIdentity,
    ) -> dict[str, str | int] | None:
        point_id = _normalize_point_id(INDEX_METADATA_LOGICAL_POINT_ID)
        try:
            points = self._probe_client.retrieve(
                _physical_index_name(target),
                (point_id,),
                with_payload=True,
                with_vectors=False,
            )
        except (OSError, ConnectionError, TimeoutError):
            return None
        if not points:
            return None
        return _extract_provider_payload(points[0].payload)

    def _read_embedding_probe_payload(
        self,
        target: VectorIndexIdentity,
    ) -> dict[str, str | int] | None:
        try:
            points, _offset = self._probe_client.scroll(
                _physical_index_name(target),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
        except (OSError, ConnectionError, TimeoutError):
            return None
        if not points:
            return None
        return _extract_provider_payload(points[0].payload)


__all__ = [
    "QdrantIndexProbeClient",
    "QdrantVectorIndexIdentityResolver",
]
