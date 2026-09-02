"""Qdrant reference search-index bootstrap adapter — vendor imports isolated here."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.vector_store.qdrant.config import QdrantIntegrationConfig
from intergrax.rag.vectorstore.sparse.sparse_encoder import SparseEncoder, resolve_sparse_encoder

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.errors import (
    VpiBootstrapCompatibilityError,
    VpiBootstrapProviderError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.ports import (
    SearchIndexIngestBatch,
    SearchIndexIngestBatchResult,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    ValidationCheck,
    ValidationReport,
    ValidationStatus,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.deterministic_ids import (
    source_ref_payload,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.model import (
    VpiBootstrapManifest,
)
from platform_proofs.scenarios.verified_product_identification.integrations.search_store.qdrant.client_protocol import (
    QdrantBootstrapClient,
)
from platform_proofs.scenarios.verified_product_identification.integrations.search_store.qdrant.collection_compat import (
    collection_dense_dimension,
    collection_has_sparse_channel,
    is_collection_not_found,
)

_DENSE_VECTOR_NAME = "dense"
_SPARSE_VECTOR_NAME = "sparse"
_LOGICAL_ID_METADATA_KEY = "logical_id"


def _load_qdrant_client_class():
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise IntegrationConfigurationError(
            "Qdrant integration requires qdrant-client. "
            "Install with: Intergrax-ai[vector-qdrant]."
        ) from exc
    return QdrantClient


def _normalize_point_id(raw_id: str) -> str:
    try:
        return str(uuid.UUID(raw_id))
    except ValueError:
        pass
    if raw_id.isdigit():
        return raw_id
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id))


@dataclass(slots=True)
class QdrantSearchIndexBootstrapAdapter:
    """Reference ``SearchIndexBootstrapPort`` with independent lexical+dense channels."""

    _client: QdrantBootstrapClient
    _collection_name: str
    _tenant_id: str
    _sparse_enabled: bool
    _sparse_encoder: SparseEncoder
    _dimension: int | None = None

    @classmethod
    def from_env(
        cls,
        *,
        collection_name: str,
        tenant_id: str | None = None,
        enable_sparse_vectors: bool = True,
    ) -> QdrantSearchIndexBootstrapAdapter:
        config = QdrantIntegrationConfig.from_env(collection_name=collection_name)
        resolved_tenant = tenant_id or config.tenant_id
        QdrantClient = _load_qdrant_client_class()
        if config.resolved_url():
            client = QdrantClient(url=config.resolved_url(), api_key=config.api_key or None)
        else:
            client = QdrantClient(host=config.host, port=config.port, api_key=config.api_key or None)

        return cls(
            _client=client,
            _collection_name=f"{collection_name}__tenant__{resolved_tenant}",
            _tenant_id=resolved_tenant,
            _sparse_enabled=enable_sparse_vectors,
            _sparse_encoder=resolve_sparse_encoder(),
        )

    def probe_readiness(self) -> ValidationReport:
        try:
            self._client.get_collections()
        except Exception as exc:
            raise VpiBootstrapProviderError("Qdrant readiness probe failed") from exc
        return ValidationReport.from_checks(
            (
                ValidationCheck(
                    name="qdrant_reachable",
                    status=ValidationStatus.PASS,
                    detail="get_collections succeeded",
                ),
            )
        )

    def prepare(self, manifest: VpiBootstrapManifest) -> ValidationReport:
        self._dimension = manifest.embedding_dimension
        info = self._get_collection_info()
        if info is None:
            self._create_collection(dimension=manifest.embedding_dimension)
            return ValidationReport.from_checks(
                (
                    ValidationCheck(
                        name="qdrant_collection_created",
                        status=ValidationStatus.PASS,
                        detail=f"collection={self._collection_name}",
                    ),
                )
            )

        existing_dim = collection_dense_dimension(info, dense_vector_name=_DENSE_VECTOR_NAME)
        if existing_dim is None:
            raise VpiBootstrapCompatibilityError(
                f"Qdrant collection {self._collection_name!r} has no dense vector config; "
                "explicit rebuild required"
            )
        if existing_dim != manifest.embedding_dimension:
            raise VpiBootstrapCompatibilityError(
                f"Qdrant collection dimension {existing_dim} != expected "
                f"{manifest.embedding_dimension}; explicit rebuild required"
            )
        if self._sparse_enabled and not collection_has_sparse_channel(
            info,
            sparse_vector_name=_SPARSE_VECTOR_NAME,
        ):
            raise VpiBootstrapCompatibilityError(
                f"Qdrant collection {self._collection_name!r} is dense-only but bootstrap "
                "requires lexical sparse channel; explicit rebuild required"
            )
        return ValidationReport.from_checks(
            (
                ValidationCheck(
                    name="qdrant_collection_compatible",
                    status=ValidationStatus.PASS,
                    detail=(
                        f"collection={self._collection_name} dimension={existing_dim} "
                        f"sparse={self._sparse_enabled}"
                    ),
                ),
            )
        )

    def ingest_batch(self, batch: SearchIndexIngestBatch) -> SearchIndexIngestBatchResult:
        if self._dimension is None:
            raise VpiBootstrapProviderError("search index prepare must run before ingest")

        from qdrant_client.http.models import PointStruct

        points: list[PointStruct] = []
        for record in batch.records:
            payload: dict[str, str | int | None] = {
                _LOGICAL_ID_METADATA_KEY: record.logical_point_id,
                "text": record.lexical_text,
                "channel_lexical": record.lexical_text,
                "channel_vector_source": "semantic",
                "search_representation_derivation_version": record.derivation_version,
                "dataset_checksum": record.dataset_checksum,
                "embedding_provider": record.embedding_provider,
                "embedding_model": record.embedding_model,
                "embedding_dimension": record.embedding_dimension,
                **source_ref_payload(record.source_ref),
            }
            point_id = _normalize_point_id(record.logical_point_id)
            if self._sparse_enabled:
                from qdrant_client.http.models import SparseVector as QdrantSparseVector

                sparse = self._sparse_encoder.encode(record.lexical_text)
                vector = {
                    _DENSE_VECTOR_NAME: list(record.dense_embedding),
                    _SPARSE_VECTOR_NAME: QdrantSparseVector(
                        indices=sparse.indices,
                        values=sparse.values,
                    ),
                }
            else:
                vector = list(record.dense_embedding)
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        try:
            self._client.upsert(collection_name=self._collection_name, points=points)
        except Exception as exc:
            raise VpiBootstrapProviderError(
                f"Qdrant ingest failed for batch {batch.batch_ordinal}"
            ) from exc
        return SearchIndexIngestBatchResult(point_count=self.count_points())

    def validate(self, manifest: VpiBootstrapManifest) -> ValidationReport:
        point_count = self.count_points()
        from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.run_target import (
            effective_run_target_rows,
        )

        expected = effective_run_target_rows(manifest)
        return ValidationReport.from_checks(
            (
                ValidationCheck(
                    name="qdrant_point_count",
                    status=ValidationStatus.PASS if point_count >= expected else ValidationStatus.FAIL,
                    detail=f"point_count={point_count} expected>={expected}",
                ),
                ValidationCheck(
                    name="qdrant_dimension",
                    status=(
                        ValidationStatus.PASS
                        if self._dimension == manifest.embedding_dimension
                        else ValidationStatus.FAIL
                    ),
                    detail=f"configured_dimension={self._dimension}",
                ),
            )
        )

    def count_points(self) -> int:
        info = self._get_collection_info()
        if info is None:
            return 0
        return int(info.points_count)

    def close(self) -> None:
        self._client.close()

    def _get_collection_info(self):
        try:
            return self._client.get_collection(self._collection_name)
        except Exception as exc:
            if is_collection_not_found(exc):
                return None
            raise VpiBootstrapProviderError("Qdrant get_collection failed") from exc

    def _create_collection(self, *, dimension: int) -> None:
        from qdrant_client.http.models import Distance, SparseIndexParams, SparseVectorParams, VectorParams

        metric = Distance.COSINE
        if self._sparse_enabled:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config={
                    _DENSE_VECTOR_NAME: VectorParams(size=dimension, distance=metric),
                },
                sparse_vectors_config={
                    _SPARSE_VECTOR_NAME: SparseVectorParams(index=SparseIndexParams()),
                },
            )
            return
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=dimension, distance=metric),
        )
