"""Qdrant vector storage bootstrap adapter for Data Pack load."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from qdrant_client.http.models import Distance, PointStruct

from intergrax.integrations.contracts.vector_index_administration import (
    VectorIndexAdministration,
    VectorIndexCompatibilityError,
)
from intergrax.integrations.providers.vector_store.qdrant.index_administration import (
    build_qdrant_index_spec,
)
from intergrax.integrations.providers.vector_store.qdrant.opens import (
    _build_qdrant_client,
    open_qdrant_vector_index_administration,
)
from intergrax.integrations.providers.vector_store.qdrant.rag_store import (
    _normalize_point_id,
)

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.configuration import (
    ExpectedVectorIdentity,
    QdrantBootstrapConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.errors import (
    QdrantBootstrapCollectionError,
    QdrantBootstrapConfigurationError,
    QdrantBootstrapIdentityConflictError,
    QdrantBootstrapOperationError,
    QdrantBootstrapVectorValidationError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.payload import (
    QdrantStoredPoint,
    QdrantUpsertPoint,
    QdrantVectorPayload,
    embedding_identity_matches,
    normalize_vector_float32,
    payload_from_provider_dict,
    payload_from_record,
    payload_identity_matches,
    vectors_transport_equal,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.target_mapping import (
    PhysicalVectorTarget,
    resolve_physical_target,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    StorageLoadBatchResult,
    VectorBatch,
    VectorLoadRecord,
    VectorTargetId,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.errors import (
    StorageBootstrapIntegrityError,
    StorageBootstrapWriteError,
)


type QdrantProviderDistance = str | Distance
type QdrantProviderPayloadInput = Mapping[str, str | int] | None
type QdrantProviderVectorInput = list[float] | dict[str, list[float]] | None


class QdrantVectorParamsView(Protocol):
    size: int
    distance: QdrantProviderDistance


class QdrantCollectionParamsView(Protocol):
    vectors: QdrantVectorParamsView | dict[str, QdrantVectorParamsView] | None


class QdrantCollectionConfigView(Protocol):
    params: QdrantCollectionParamsView


class QdrantCollectionInfoView(Protocol):
    config: QdrantCollectionConfigView


class QdrantUpsertPointView(Protocol):
    id: str | int
    vector: list[float] | dict[str, list[float]]
    payload: dict[str, str | int]


class QdrantProviderPoint(Protocol):
    id: str | int
    payload: QdrantProviderPayloadInput
    vector: QdrantProviderVectorInput


class QdrantDataPlaneClient(Protocol):
    def retrieve(
        self,
        collection_name: str,
        ids: Sequence[str | int],
        *,
        with_payload: bool,
        with_vectors: bool,
    ) -> Sequence[QdrantProviderPoint]: ...

    def upsert(
        self,
        collection_name: str,
        points: Sequence[QdrantUpsertPointView],
    ) -> None: ...

    def get_collection(self, collection_name: str) -> QdrantCollectionInfoView: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _CollectionVectorShape:
    dimension: int
    distance: str
    uses_named_dense_vector: bool
    dense_vector_channel_name: str | None


def _sanitize_provider_error(exc: BaseException) -> str:
    return f"qdrant operation failed: {type(exc).__name__}"


def _validate_vector_record(
    record: VectorLoadRecord,
    expected_identity: ExpectedVectorIdentity,
) -> None:
    if not embedding_identity_matches(record, expected_identity):
        raise QdrantBootstrapVectorValidationError("VECTOR_IDENTITY_INCOMPATIBLE")
    for value in record.dense_embedding:
        if not math.isfinite(value):
            raise QdrantBootstrapVectorValidationError("non-finite vector value")
    norm = math.sqrt(sum(value * value for value in record.dense_embedding))
    if norm <= 0.0:
        raise QdrantBootstrapVectorValidationError("zero vector rejected")


def _extract_provider_payload(raw_payload: QdrantProviderPayloadInput) -> dict[str, str | int]:
    if raw_payload is None:
        raise QdrantBootstrapOperationError("stored payload missing")
    converted: dict[str, str | int] = {}
    for key, value in raw_payload.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, str) or isinstance(value, int):
            converted[key] = value
    return converted


def _extract_dense_vector(
    raw_vector: QdrantProviderVectorInput,
    physical: PhysicalVectorTarget,
) -> tuple[float, ...]:
    if physical.uses_named_dense_vector:
        if not isinstance(raw_vector, dict):
            raise QdrantBootstrapOperationError("stored vector missing named dense channel")
        channel_vector = raw_vector.get(physical.dense_vector_channel_name)
        if channel_vector is None:
            raise QdrantBootstrapOperationError("stored vector missing named dense channel")
        if not isinstance(channel_vector, list):
            raise QdrantBootstrapOperationError("stored dense vector has invalid type")
        return tuple(float(value) for value in channel_vector)
    if isinstance(raw_vector, dict):
        raise QdrantBootstrapOperationError("stored vector uses unexpected named channels")
    if not isinstance(raw_vector, list):
        raise QdrantBootstrapOperationError("stored vector has invalid type")
    return tuple(float(value) for value in raw_vector)


def _stored_point_from_provider_record(
    provider_point: QdrantProviderPoint,
    physical: PhysicalVectorTarget,
) -> QdrantStoredPoint:
    payload = payload_from_provider_dict(_extract_provider_payload(provider_point.payload))
    vector = _extract_dense_vector(provider_point.vector, physical)
    return QdrantStoredPoint(
        point_id=provider_point.id,
        logical_point_id=payload.logical_point_id,
        payload=payload,
        vector=vector,
    )


def _distance_label(distance: QdrantProviderDistance) -> str:
    return str(distance)


def _collection_vector_shape(
    collection_info: QdrantCollectionInfoView,
    physical: PhysicalVectorTarget,
) -> _CollectionVectorShape:
    vectors = collection_info.config.params.vectors
    if vectors is None:
        raise QdrantBootstrapCollectionError("collection has no dense vector config")
    if physical.uses_named_dense_vector:
        if not isinstance(vectors, dict):
            raise QdrantBootstrapCollectionError("expected named dense vector channel")
        dense = vectors.get(physical.dense_vector_channel_name)
        if dense is None:
            raise QdrantBootstrapCollectionError("expected named dense vector channel")
        return _CollectionVectorShape(
            dimension=int(dense.size),
            distance=_distance_label(dense.distance),
            uses_named_dense_vector=True,
            dense_vector_channel_name=physical.dense_vector_channel_name,
        )
    if isinstance(vectors, dict):
        raise QdrantBootstrapCollectionError("collection uses named vectors but adapter expects default dense")
    return _CollectionVectorShape(
        dimension=int(vectors.size),
        distance=_distance_label(vectors.distance),
        uses_named_dense_vector=False,
        dense_vector_channel_name=None,
    )


def _distance_is_cosine(distance: str) -> bool:
    normalized = distance.strip().lower()
    return normalized in {"cosine", "distance.cosine"}


def _record_matches_stored(
    record: VectorLoadRecord,
    stored: QdrantStoredPoint,
    *,
    tolerance: float,
) -> bool:
    expected_payload = payload_from_record(record)
    if not payload_identity_matches(stored.payload, expected_payload):
        return False
    expected_vector = normalize_vector_float32(record.dense_embedding)
    return vectors_transport_equal(expected_vector, stored.vector, tolerance=tolerance)


@dataclass(slots=True)
class QdrantVectorStorageAdapter:
    """``VectorStorageLoadPort`` implementation over platform Qdrant primitives."""

    _client: QdrantDataPlaneClient
    _index_admin: VectorIndexAdministration
    _configuration: QdrantBootstrapConfiguration
    _prepared_targets: set[str]

    @classmethod
    def from_env(
        cls,
        *,
        logical_collection_name: str | None = None,
    ) -> QdrantVectorStorageAdapter:
        if logical_collection_name is None:
            configuration = QdrantBootstrapConfiguration.from_env()
        else:
            configuration = QdrantBootstrapConfiguration.from_env(
                logical_collection_name=logical_collection_name,
            )
        client = _build_qdrant_client(configuration.integration)
        index_admin = open_qdrant_vector_index_administration(configuration.integration)
        return cls(
            _client=client,
            _index_admin=index_admin,
            _configuration=configuration,
            _prepared_targets=set(),
        )

    def close(self) -> None:
        self._index_admin.close()
        self._client.close()

    def prepare_target(self, logical_target: VectorTargetId) -> None:
        physical = resolve_physical_target(logical_target, self._configuration)
        expected = self._configuration.expected_vector_identity
        spec = build_qdrant_index_spec(
            identity=physical.index_identity,
            dimension=expected.dimension,
            metric="cosine",
            enable_sparse_lexical=False,
            dense_channel_name=self._configuration.dense_vector_channel_name,
        )
        try:
            self._index_admin.prepare_index(spec)
        except VectorIndexCompatibilityError as exc:
            raise QdrantBootstrapCollectionError(str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise QdrantBootstrapCollectionError("qdrant prepare_target failed") from exc

        try:
            shape = _collection_vector_shape(
                self._client.get_collection(physical.collection_name),
                physical,
            )
        except QdrantBootstrapCollectionError:
            raise
        except (OSError, ValueError) as exc:
            raise QdrantBootstrapCollectionError("qdrant collection verification failed") from exc

        if shape.dimension != expected.dimension:
            raise QdrantBootstrapCollectionError(
                f"collection vector size {shape.dimension} != expected {expected.dimension}"
            )
        if not _distance_is_cosine(shape.distance):
            raise QdrantBootstrapCollectionError(
                f"collection distance {shape.distance!r} is not cosine"
            )
        self._prepared_targets.add(str(logical_target))

    def write_batch(self, batch: VectorBatch) -> StorageLoadBatchResult:
        physical = resolve_physical_target(batch.target, self._configuration)
        requested = len(batch.records)
        if requested == 0:
            return StorageLoadBatchResult(
                requested_count=0,
                written_count=0,
                updated_count=0,
                skipped_count=0,
                failed_count=0,
            )
        if str(batch.target) not in self._prepared_targets:
            self.prepare_target(batch.target)

        tolerance = self._configuration.vector_transport_tolerance
        expected_identity = self._configuration.expected_vector_identity

        written_count = 0
        skipped_count = 0
        first_failed_identity: str | None = None

        try:
            for record in batch.records:
                _validate_vector_record(record, expected_identity)
            existing_by_logical = self._retrieve_existing_points(batch.records, physical)
            to_write: list[VectorLoadRecord] = []
            for record in batch.records:
                stored = existing_by_logical.get(record.logical_point_id)
                if stored is None:
                    to_write.append(record)
                    continue
                if _record_matches_stored(record, stored, tolerance=tolerance):
                    skipped_count += 1
                    continue
                first_failed_identity = record.logical_point_id
                raise QdrantBootstrapIdentityConflictError(
                    f"VECTOR_CONTENT_CONFLICT: {record.logical_point_id}"
                )

            if to_write:
                self._upsert_records(to_write, physical)
                written_count = len(to_write)

            verify_result = self._verify_records(batch.records, physical)
            if verify_result.failed_count > 0:
                raise StorageBootstrapIntegrityError("post-write vector batch verification failed")
        except QdrantBootstrapIdentityConflictError as exc:
            raise StorageBootstrapWriteError(str(exc)) from exc
        except QdrantBootstrapVectorValidationError as exc:
            raise StorageBootstrapWriteError(str(exc)) from exc
        except QdrantBootstrapConfigurationError as exc:
            raise StorageBootstrapWriteError(str(exc)) from exc
        except QdrantBootstrapCollectionError as exc:
            raise StorageBootstrapWriteError(str(exc)) from exc
        except QdrantBootstrapOperationError as exc:
            raise StorageBootstrapWriteError(str(exc)) from exc
        except StorageBootstrapIntegrityError:
            raise
        except (OSError, ValueError) as exc:
            raise StorageBootstrapWriteError("Qdrant vector batch write failed") from exc

        return StorageLoadBatchResult(
            requested_count=requested,
            written_count=written_count,
            updated_count=0,
            skipped_count=skipped_count,
            failed_count=0,
            first_failed_identity=first_failed_identity,
        )

    def verify_batch(self, batch: VectorBatch) -> StorageLoadBatchResult:
        physical = resolve_physical_target(batch.target, self._configuration)
        requested = len(batch.records)
        if requested == 0:
            return StorageLoadBatchResult(
                requested_count=0,
                written_count=0,
                updated_count=0,
                skipped_count=0,
                failed_count=0,
            )
        try:
            return self._verify_records(batch.records, physical)
        except QdrantBootstrapOperationError as exc:
            raise StorageBootstrapIntegrityError(str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise StorageBootstrapIntegrityError(
                "Qdrant vector batch verification failed"
            ) from exc

    def _verify_records(
        self,
        records: Sequence[VectorLoadRecord],
        physical: PhysicalVectorTarget,
    ) -> StorageLoadBatchResult:
        tolerance = self._configuration.vector_transport_tolerance
        existing_by_logical = self._retrieve_existing_points(records, physical)
        verified = 0
        first_failed_identity: str | None = None
        for record in records:
            stored = existing_by_logical.get(record.logical_point_id)
            if stored is not None and _record_matches_stored(record, stored, tolerance=tolerance):
                verified += 1
            elif first_failed_identity is None:
                first_failed_identity = record.logical_point_id
        failed = len(records) - verified
        return StorageLoadBatchResult(
            requested_count=len(records),
            written_count=verified,
            updated_count=0,
            skipped_count=0,
            failed_count=failed,
            first_failed_identity=first_failed_identity,
        )

    def _retrieve_existing_points(
        self,
        records: Sequence[VectorLoadRecord],
        physical: PhysicalVectorTarget,
    ) -> dict[str, QdrantStoredPoint]:
        if not records:
            return {}
        point_ids = [_normalize_point_id(record.logical_point_id) for record in records]
        logical_by_point_id = {
            _normalize_point_id(record.logical_point_id): record.logical_point_id
            for record in records
        }
        stored: dict[str, QdrantStoredPoint] = {}
        batch_size = self._configuration.upsert_batch_size
        for start in range(0, len(point_ids), batch_size):
            chunk_ids = point_ids[start : start + batch_size]
            try:
                provider_points = self._client.retrieve(
                    physical.collection_name,
                    chunk_ids,
                    with_payload=True,
                    with_vectors=True,
                )
            except (OSError, ValueError) as exc:
                raise QdrantBootstrapOperationError(_sanitize_provider_error(exc)) from exc
            for provider_point in provider_points:
                converted = _stored_point_from_provider_record(provider_point, physical)
                logical_id = logical_by_point_id.get(converted.point_id, converted.logical_point_id)
                stored[logical_id] = converted
        return stored

    def _upsert_records(
        self,
        records: Sequence[VectorLoadRecord],
        physical: PhysicalVectorTarget,
    ) -> None:
        batch_size = self._configuration.upsert_batch_size
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            points = [
                self._to_sdk_upsert_point(self._build_provider_point(record, physical))
                for record in chunk
            ]
            try:
                self._client.upsert(physical.collection_name, points)
            except (OSError, ValueError) as exc:
                raise QdrantBootstrapOperationError(_sanitize_provider_error(exc)) from exc

    def _build_provider_point(
        self,
        record: VectorLoadRecord,
        physical: PhysicalVectorTarget,
    ) -> QdrantUpsertPoint:
        payload = payload_from_record(record).to_provider_payload()
        vector_values = list(normalize_vector_float32(record.dense_embedding))
        point_id = _normalize_point_id(record.logical_point_id)
        if physical.uses_named_dense_vector:
            vector_payload: list[float] | dict[str, list[float]] = {
                physical.dense_vector_channel_name: vector_values
            }
        else:
            vector_payload = vector_values
        return QdrantUpsertPoint(id=point_id, vector=vector_payload, payload=payload)

    def _to_sdk_upsert_point(self, point: QdrantUpsertPoint) -> PointStruct | QdrantUpsertPoint:
        try:
            from qdrant_client.http.models import PointStruct
        except ImportError:
            return point
        return PointStruct(id=point.id, vector=point.vector, payload=point.payload)
