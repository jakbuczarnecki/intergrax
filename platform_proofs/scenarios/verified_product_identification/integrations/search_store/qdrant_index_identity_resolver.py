"""Qdrant-backed resolver for vector index runtime identity."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.integrations.contracts.base import IntegrationDependencyError
from intergrax.integrations.contracts.vector_index_administration import (
    VectorIndexAdministration,
    VectorIndexIdentity,
)
from intergrax.integrations.contracts.vector_index_metadata import VectorIndexMetadataReader
from intergrax.integrations.providers.vector_store.qdrant.config import QdrantIntegrationConfig
from intergrax.integrations.providers.vector_store.qdrant.opens import (
    open_qdrant_vector_index_administration,
    open_qdrant_vector_index_metadata_reader,
)

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


@dataclass(slots=True)
class QdrantVectorIndexIdentityResolver:
    """Resolve Qdrant index identity via public administration and metadata seams."""

    _index_admin: VectorIndexAdministration
    _metadata_reader: VectorIndexMetadataReader

    @classmethod
    def from_qdrant_config(
        cls,
        config: QdrantIntegrationConfig,
        *,
        index_admin: VectorIndexAdministration | None = None,
        metadata_reader: VectorIndexMetadataReader | None = None,
    ) -> QdrantVectorIndexIdentityResolver:
        resolved_admin = (
            index_admin
            if index_admin is not None
            else open_qdrant_vector_index_administration(config)
        )
        resolved_reader = (
            metadata_reader
            if metadata_reader is not None
            else open_qdrant_vector_index_metadata_reader(config)
        )
        return cls(
            _index_admin=resolved_admin,
            _metadata_reader=resolved_reader,
        )

    def close(self) -> None:
        self._index_admin.close()
        self._metadata_reader.close()

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
        if description.dense_dimension is None:
            return VectorIndexIdentityResolution(
                status=VectorIndexIdentityResolutionStatus.METADATA_UNAVAILABLE,
                identity=None,
            )
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
                    dimension=description.dense_dimension,
                    metric=description.dense_metric,
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
                dimension=description.dense_dimension,
                metric=description.dense_metric,
                content_identity=None,
            ),
        )

    def _read_index_metadata_payload(
        self,
        target: VectorIndexIdentity,
    ) -> dict[str, str | int] | None:
        record = self._metadata_reader.retrieve_point_by_logical_id(
            target,
            INDEX_METADATA_LOGICAL_POINT_ID,
        )
        if record is None:
            return None
        return record.payload

    def _read_embedding_probe_payload(
        self,
        target: VectorIndexIdentity,
    ) -> dict[str, str | int] | None:
        record = self._metadata_reader.retrieve_first_point_payload(target, limit=1)
        if record is None:
            return None
        return record.payload


__all__ = [
    "QdrantVectorIndexIdentityResolver",
]
